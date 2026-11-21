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


def _extraction_prompt(chunk: Chunk, codes: set[str]) -> str:
    if "## Состав узлов" in chunk.text:
        part_count = sum(1 for code in codes if code_type(code) == "Part")
        return f"""Извлеки сущности и связи из ведомости деталей. Схема направлений:
{_DIRECTIONS}
В тексте {part_count} артикулов. Заполни relations по группам:
1. BELONGS_TO от модели к линейке.
2. Все {part_count} связей HAS_PART. Один и тот же Model из заголовка слева, разные Part справа.
3. Все {part_count} связей IS_A. Для каждого артикула возьми название узла из его строки.
4. SUPPLIED_BY от каждого артикула к поставщику из той же строки.
Пример для строки «контроллер — деталь P-9001, поставщик Завод» в руководстве GS-900:
GS-900 HAS_PART P-9001; P-9001 IS_A контроллер; P-9001 SUPPLIED_BY Завод.
В entities перечисли концы всех связей. Используй только сведения из документа."""
    if chunk.doc_id.startswith("report_"):
        return f"""Извлеки сущности и связи из отчёта о выезде. Схема направлений:
{_DIRECTIONS}
Для фактического отказа выпиши Model, ErrorCode и Component. Скопируй структуру примера,
заменив значения полей subject и object на найденные в документе:
{{"subject_type":"Model","subject":"GS-900","type":"FAILED","object_type":"Component","object":"контроллер"}}
{{"subject_type":"ErrorCode","subject":"E-90","type":"INDICATES","object_type":"Component","object":"контроллер"}}
В entities перечисли все сущности. Номер отчёта и город пропусти."""
    if chunk.doc_id.startswith("bulletin_"):
        return f"""Извлеки из бюллетеня сведения о деталях. Схема направлений:
{_DIRECTIONS}
Пример: «Деталь P-9001 (контроллер) снята, новая деталь P-9002. Её поставщик Завод».
Записи в relations по этому примеру:
{{"subject_type":"Part","subject":"P-9001","type":"IS_A","object_type":"Component","object":"контроллер"}}
{{"subject_type":"Part","subject":"P-9001","type":"REPLACED_BY","object_type":"Part","object":"P-9002"}}
{{"subject_type":"Part","subject":"P-9002","type":"SUPPLIED_BY","object_type":"Supplier","object":"Завод"}}
Если поставщик сообщил о дефекте партии одного артикула, выпиши только IS_A и SUPPLIED_BY для этого артикула.
В entities и relations копируй полные имена и артикулы из текста, включая последние цифры и буквы."""
    if chunk.doc_id.startswith("procedure_"):
        return f"""Извлеки из регламента сведения о диагностике и ремонте. Схема направлений:
{_DIRECTIONS}
Пример: регламент SP-901 устраняет ошибку E-90, указывающую на отказ контроллера.
{{"subject_type":"ErrorCode","subject":"E-90","type":"FIXED_BY","object_type":"Procedure","object":"SP-901"}}
{{"subject_type":"ErrorCode","subject":"E-90","type":"INDICATES","object_type":"Component","object":"контроллер"}}
Обе связи начинаются с кода ошибки, выпиши его целиком. В entities включи концы связей.
Если регламент описывает только плановые работы без ошибок, верни {{"entities":[],"relations":[]}}."""
    return f"""Извлеки явно описанные сущности и связи. Схема направлений:
{_DIRECTIONS}
В таблице кодов каждая строка даёт связь ErrorCode INDICATES Component.
Пример: «E-90 — отказ узла контроллер» даёт E-90 INDICATES контроллер.
Перечисли все факты из документа, точно перепиши коды и названия в relations и entities."""


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
    instruction = _extraction_prompt(chunk, codes)
    messages = [
        {"role": "system", "content": instruction},
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
