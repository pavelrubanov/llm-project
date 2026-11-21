"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from pydantic import BaseModel, ValidationError

from assistant.infra import schemas
from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult


class _EntityRow(BaseModel):
    type: str
    name: str


class _RelationRow(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class _Extraction(BaseModel):
    entities: list[_EntityRow]
    relations: list[_RelationRow]


directions = []
for kind, (source, target) in schemas.RELATION_SIGNATURES.items():
    directions.append(f"{source} -{kind}-> {target}")
_DIRECTIONS = "\n".join(directions)
_PROMPT = """Извлеки все явно описанные сущности и связи из одного фрагмента сервисного документа.
Допустимые направления связей:
""" + _DIRECTIONS + """
В перечне установленных узлов обработай КАЖДУЮ строку: модель HAS_PART деталь,
деталь IS_A узел и деталь SUPPLIED_BY поставщик. Линейку свяжи с моделью через BELONGS_TO.
В таблице ошибок каждая ошибка INDICATES названный неисправный узел.
В регламенте ошибка FIXED_BY регламент. FAILED извлекай только при фактически произошедшем отказе модели.
В бюллетене направление REPLACED_BY идёт от снятой детали к её замене, даже если первой названа новая.
Дефект партии сам по себе не означает замену. Не придумывай коды, не превращай номера отчётов в сущности.
Названия узлов приводи к именительному падежу. Включи концы каждой связи в entities.
Пример: «В AR-900 стоит контроллер P-9000» даёт Model AR-900, Component контроллер, Part P-9000 и две связи:
{"subject_type":"Model","subject":"AR-900","type":"HAS_PART","object_type":"Part","object":"P-9000"}
{"subject_type":"Part","subject":"P-9000","type":"IS_A","object_type":"Component","object":"контроллер"}.
Пример задаёт формат. Верни JSON только о тексте пользователя."""


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    codes = set(find_codes(chunk.text))
    vocab = vocabulary()
    hints = [(code_type(code), code) for code in sorted(codes)]
    hints.extend(find_names(chunk.text, vocab))
    names = ", ".join(f"{kind}={name}" for kind, name in hints)
    messages = [
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": f"Варианты написания обнаруженных имён: {names}\n\nДокумент:\n{chunk.text}"},
    ]
    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    retry_prompt = "Повтори извлечение целиком в формате заданной JSON-схемы."
    for attempt in range(2):
        response = llm.chat(messages, json_schema=schemas.EXTRACTION_SCHEMA, max_tokens=4096)
        try:
            rows = _Extraction.model_validate_json(response.content, strict=True)
            break
        except ValidationError:
            if attempt == 0:
                messages.append({"role": "user", "content": retry_prompt})
    else:
        return result

    def grounded(kind: str, raw: str) -> schemas.Entity | None:
        name = canonical_name(kind, raw, vocab)
        if name is None:
            return None
        if kind in CODE_TYPES and name not in codes:
            return None
        return schemas.Entity(kind, name)

    entities: set[schemas.Entity] = set()
    relations: set[schemas.Relation] = set()
    for row in rows.entities:
        entity = grounded(row.type, row.name)
        if entity:
            entities.add(entity)
        else:
            result.dropped += 1
    for edge in rows.relations:
        subject = grounded(edge.subject_type, edge.subject)
        obj = grounded(edge.object_type, edge.object)
        if subject is None or obj is None or subject == obj:
            result.dropped += 1
            continue
        relation = schemas.Relation(subject, edge.type, obj)
        if not relation.matches_schema():
            result.dropped += 1
            continue
        entities.update((subject, obj))
        relations.add(relation)
    result.entities = sorted(entities, key=lambda e: e.node_id)
    result.relations = sorted(relations, key=lambda r: (r.subject.node_id, r.type, r.object.node_id))
    return result
