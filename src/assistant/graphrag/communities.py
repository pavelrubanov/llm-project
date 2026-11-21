"""Сообщества графа и их сводки. Требование FR-3."""

import re
from collections import Counter

import networkx as nx

from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

_RELATION = {
    "BELONGS_TO": "входит в линейку",
    "HAS_PART": "содержит деталь",
    "IS_A": "имеет назначение",
    "INDICATES": "указывает на узел",
    "FIXED_BY": "устраняется по регламенту",
    "REPLACED_BY": "заменена на",
    "SUPPLIED_BY": "поставляется компанией",
}


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    undirected = graph.to_undirected()
    groups = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    return [Community(number, sorted(nodes)) for number, nodes in enumerate(groups)]


def _line_statistics(line: str, graph: nx.DiGraph) -> tuple[list[str], set[str]]:
    by_model = Counter()
    by_component = Counter()
    discontinued = set()
    chunks = set()
    for model, _, membership in graph.in_edges(line, data=True):
        if membership["type"] != "BELONGS_TO":
            continue
        chunks.update(membership["sources"])
        for target, edge in graph[model].items():
            name = graph.nodes[target]["name"]
            if edge["type"] == "FAILED":
                by_model[graph.nodes[model]["name"]] += edge["weight"]
                by_component[name] += edge["weight"]
                chunks.update(edge["sources"])
            if edge["type"] == "HAS_PART":
                for successor in graph[target].values():
                    if successor["type"] == "REPLACED_BY":
                        discontinued.add(name)
                        chunks.update(edge["sources"])
                        chunks.update(successor["sources"])

    name = graph.nodes[line]["name"]
    facts = []
    for label, counts in [("узлам", by_component), ("моделям", by_model)]:
        if counts:
            values = ", ".join(f"{item} — {n}" for item, n in counts.most_common())
            facts.append(f"Отказы по {label} линейки «{name}»: {values}.")
    if discontinued:
        parts = ", ".join(sorted(discontinued))
        facts.append(f"Снятые с производства детали из руководств линейки «{name}»: {parts}.")
    return facts, chunks


def community_facts(community: Community, graph: nx.DiGraph) -> tuple[list[str], set[str]]:
    """Факты для сводки и документы, подтверждающие их."""
    facts = []
    statistics = []
    chunks = set()
    for node in community.nodes:
        name = graph.nodes[node]["name"]
        if graph.nodes[node]["type"] == "ProductLine":
            line_facts, line_chunks = _line_statistics(node, graph)
            statistics.extend(line_facts)
            chunks.update(line_chunks)
        for target, edge in graph[node].items():
            other = graph.nodes[target]["name"]
            kind = edge["type"]
            if kind == "FAILED":
                fact = f'{name}, {other}: зарегистрировано отказов — {edge["weight"]}.'
            else:
                fact = f"{name} {_RELATION[kind]} {other}."
            facts.append(fact)
            chunks.update(edge["sources"])
        if not graph[node]:
            facts.append(f'{graph.nodes[node]["type"]} — {name}.')
            chunks.update(graph.nodes[node]["sources"])
    sources = {chunk.split("#")[0] for chunk in chunks}
    return statistics + facts, sources


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    facts, _ = community_facts(community, graph)
    context = "\n".join(facts)
    instruction = (
        "Сократи сервисные записи до 150 слов. Строки с итогами по линейке сохрани с точными названиями и числами. "
        "Укажи коды и направления замен, если они есть. "
        "Используй только приведённые записи. Не добавляй проценты, объяснения причин или рекомендации."
    )
    reply = llm.chat([{"role": "system", "content": instruction}, {"role": "user", "content": context}])
    summary = reply.content.strip()
    # Если в пересказе появился посторонний артикул или число, берём исходные факты.
    numbers = r"(?<![\w-])\d+(?:[.,]\d+)?(?:\s*%)?(?![\w-])"
    extra_numbers = set(re.findall(numbers, summary)) - set(re.findall(numbers, context))
    extra_codes = set(find_codes(summary)) - set(find_codes(context))
    if not summary or extra_numbers or extra_codes:
        summary = context
    return " ".join(summary.split()[:150])
