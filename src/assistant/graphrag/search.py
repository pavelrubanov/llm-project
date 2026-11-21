"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.graphrag.communities import community_facts
from assistant.infra import schemas
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase

_NO_DATA = "В базе знаний не найдено."


def _local_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    undirected = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        nearby = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        return set(graph.subgraph(nearby).edges)

    edges = set()
    for start, end in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
            for left, right in pairwise(path):
                if graph.has_edge(left, right):
                    edges.add((left, right))
                if graph.has_edge(right, left):
                    edges.add((right, left))

    path_nodes = set()
    for source, target in edges:
        path_nodes.update((source, target))
    parts = {node for node in path_nodes if graph.nodes[node]["type"] == "Part"}
    while parts:
        part = parts.pop()
        for replacement, data in graph[part].items():
            edge = (part, replacement)
            if data["type"] == "REPLACED_BY" and edge not in edges:
                edges.add(edge)
                parts.add(replacement)
    return edges


def _targets(graph, nodes, relation):
    targets = set()
    for node in nodes:
        for target, data in graph[node].items():
            if data["type"] == relation:
                targets.add(target)
    return targets


def _part_chain(graph, anchors):
    parts = {node for node in anchors if graph.nodes[node]["type"] == "Part"}
    if not parts:
        models = [node for node in anchors if graph.nodes[node]["type"] == "Model"]
        components = _targets(graph, anchors, "INDICATES")
        components.update(node for node in anchors if graph.nodes[node]["type"] == "Component")
        for part in _targets(graph, models, "HAS_PART"):
            if _targets(graph, [part], "IS_A") & components:
                parts.add(part)
    if len(parts) != 1:
        return []

    chain = [parts.pop()]
    while replacements := _targets(graph, [chain[-1]], "REPLACED_BY"):
        if len(replacements) != 1:
            return []
        replacement = replacements.pop()
        if replacement in chain:
            return []
        chain.append(replacement)
    return chain


def _answer(text: str, sources: set[str], facts: list[str]) -> Answer:
    text = text.strip()
    refusal = text.lstrip("*«").casefold().startswith("в базе знаний не найдено")
    extra_codes = set(find_codes(text)) - set(find_codes("\n".join(facts)))
    if not text or refusal or extra_codes:
        return Answer(_NO_DATA, refused=True)
    return Answer(text, sources=sorted(sources), facts=facts)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
    graph = kb.graph
    anchors = []
    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node not in graph:
            return Answer(_NO_DATA, refused=True)
        anchors.append(node)
    for kind, name in find_names(question, kb.vocabulary):
        node = f"{kind}:{name}"
        if node in graph:
            anchors.append(node)
    if not anchors:
        return Answer(_NO_DATA, refused=True)

    edges = _local_edges(graph, anchors)
    if not edges:
        return Answer(_NO_DATA, refused=True)
    text = question.casefold()
    asks_supplier = "поставля" in text or "поставщик" in text
    asks_part = any(word in text for word in ("замен", "вместо", "стави", "ставя", "заказ", "закаж", "подобр"))
    chain = _part_chain(graph, anchors) if asks_part or asks_supplier else []
    focus = set(pairwise(chain))
    suppliers = set()
    if chain and asks_supplier:
        suppliers = _targets(graph, [chain[-1]], "SUPPLIED_BY")
        focus.update((chain[-1], supplier) for supplier in suppliers)
    edges.update(focus)

    def priority(edge):
        source, target = edge
        if edge in focus:
            rank = -1
        elif graph.edges[edge]["type"] == "REPLACED_BY":
            rank = 0
        elif source in anchors or target in anchors:
            rank = 1
        else:
            rank = 2
        return rank, source, target

    facts = []
    sources = set()
    size = 0
    for source, target in sorted(edges, key=priority):
        data = graph.edges[source, target]
        left = graph.nodes[source]["name"]
        right = graph.nodes[target]["name"]
        fact = f'{len(facts) + 1}. {left} -{data["type"]}-> {right}'
        if size + len(fact) > 8000:
            break
        facts.append(fact)
        size += len(fact) + 1
        sources.update(chunk.split("#")[0] for chunk in data["sources"])
    if not facts:
        return Answer(_NO_DATA, refused=True)

    instruction = (
        "Ответь диспетчеру в одном-двух предложениях, сразу с результата. Бери сведения только из списка фактов. "
        "FIXED_BY указывает регламент, SUPPLIED_BY — поставщика. Подбирая деталь, проследи REPLACED_BY до конца. "
        "Ты даёшь справку по документам. Заказ, остатки и цена доступны только через ERP. "
        "Если ответа нет, напиши «В базе знаний не найдено.»"
    )
    context = "\n".join(facts)
    reply = llm.chat([
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Факты:\n{context}\n\nВопрос: {question}"},
    ])
    answer = _answer(reply.content, sources, facts)
    if chain:
        current = graph.nodes[chain[-1]]["name"]
        if asks_supplier and not suppliers:
            return Answer(_NO_DATA, refused=True)
        if asks_supplier and len(suppliers) == 1:
            supplier = graph.nodes[suppliers.pop()]["name"]
            named = {name for kind, name in find_names(answer.text, kb.vocabulary) if kind == "Supplier"}
            if named != {supplier} or current not in find_codes(answer.text):
                return Answer(f"{current} поставляет {supplier}.", sources=sorted(sources), facts=facts)
            return answer
        part_codes = {code for code in find_codes(answer.text) if code.startswith("P-")}
        unclear_chain = len(chain) > 2 and len(part_codes) > 1
        if asks_part and len(chain) > 1 and (unclear_chain or current not in part_codes):
            path = " → ".join(graph.nodes[node]["name"] for node in chain)
            text = f"Актуальная деталь — {current}. Цепочка замен: {path}."
            return Answer(text, sources=sorted(sources), facts=facts)
    return answer


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    partials = []
    summaries = []
    sources = set()
    map_instruction = (
        "Ответь кратко, только по сводке. Перенеси из неё точные названия, артикулы и готовые числа. "
        "Оцени полезность ответа от 0 до 10. Если сведений о запрошенном объекте нет, верни пустой ответ и score 0."
    )
    for community in kb.communities:
        messages = [
            {"role": "system", "content": map_instruction},
            {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
        ]
        reply = llm.chat(messages, json_schema=schemas.MAP_SCHEMA, max_tokens=200)
        try:
            partial = json.loads(reply.content)
        except ValueError:
            continue
        if not isinstance(partial, dict):
            continue
        score = partial.get("score")
        answer = partial.get("answer")
        if type(score) is not int or not 1 <= score <= 10:
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        partials.append(f"Частичный ответ: {answer.strip()}\nСводка-источник: {community.summary}")
        summaries.append(community.summary)
        _, documents = community_facts(community, kb.graph)
        sources.update(documents)

    if not partials:
        return Answer(_NO_DATA, refused=True)
    context = "\n\n".join(partials)
    reduce_instruction = (
        "Ответь диспетчеру одной фразой. Объедини частичные ответы, проверяя их по приложенным сводкам. "
        "Используй сводку о запрошенном объекте. Для вопроса о частоте приведи узел или модель и число отказов. "
        "Сведения о ценах, остатках и заказах доступны только через ERP. "
        "Если запрошенный факт отсутствует, напиши «В базе знаний не найдено.»"
    )
    reply = llm.chat([
        {"role": "system", "content": reduce_instruction},
        {"role": "user", "content": f"Частичные ответы:\n{context}\n\nВопрос: {question}"},
    ])
    return _answer(reply.content, sources, summaries)
