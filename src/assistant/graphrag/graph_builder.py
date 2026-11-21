"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

from collections import defaultdict

import networkx as nx

from assistant.infra import schemas
from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    node_sources: dict[schemas.Entity, set[str]] = defaultdict(set)
    edge_sources: dict[schemas.Relation, set[str]] = defaultdict(set)
    for result in results:
        for entity in result.entities:
            node_sources[entity].add(result.chunk_id)
        for relation in result.relations:
            if relation.matches_schema() and relation.subject != relation.object:
                node_sources[relation.subject].add(result.chunk_id)
                node_sources[relation.object].add(result.chunk_id)
                edge_sources[relation].add(result.chunk_id)

    graph = nx.DiGraph(conflicts=0)
    for entity, sources in node_sources.items():
        graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=sorted(sources))

    for relation, sources in edge_sources.items():
        source = relation.subject.node_id
        target = relation.object.node_id
        weight = len(sources)
        if relation.type == "REPLACED_BY":
            reverse = schemas.Relation(relation.object, relation.type, relation.subject)
            reverse_sources = edge_sources.get(reverse)
            if reverse_sources:
                # Пару увидим дважды. Считаем конфликт один раз.
                if source < target:
                    graph.graph["conflicts"] += 1
                if len(reverse_sources) >= weight:
                    continue
        graph.add_edge(source, target, type=relation.type, weight=weight, sources=sorted(sources))
    return graph
