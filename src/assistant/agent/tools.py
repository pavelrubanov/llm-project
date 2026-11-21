"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra import erp_client
from assistant.infra.erp_client import ErpClient
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_PATTERN = r"^P-[0-9]{4}[A-Z]?$"


def _schema(name: str, description: str, **properties: dict) -> dict:
    parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    function = {"name": name, "description": description, "parameters": parameters}
    return {"type": "function", "function": function}


# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = []
_question = {"type": "string", "minLength": 1}
_part = {"type": "string", "pattern": PART_PATTERN}
_warehouse = {"type": "string", "enum": WAREHOUSES}
TOOL_SCHEMAS.extend(
    [
        _schema("search_knowledge", "Найти по документам модель, ошибку, деталь или её замену.", question=_question),
        _schema("summarize_topic", "Ответить на обзорный вопрос по всему корпусу документов.", question=_question),
        _schema("check_stock", "Узнать остаток детали на выбранном складе.", part_id=_part, warehouse=_warehouse),
        _schema("get_price", "Узнать цену детали в рублях.", part_id=_part),
        _schema(
            "create_service_order",
            "Оформить заказ на деталь для модели оборудования.",
            part_id=_part,
            warehouse=_warehouse,
            qty={"type": "integer", "minimum": 1},
            model={"type": "string"},
        ),
    ]
)


def _with_validation(function: Callable, properties: dict) -> Callable:
    def call(**args) -> dict:
        if args.keys() != properties.keys():
            return {"error": "Ожидаются аргументы: " + ", ".join(properties)}
        for name, spec in properties.items():
            value = args[name]
            if spec["type"] == "string" and (not isinstance(value, str) or not value.strip()):
                return {"error": f"Параметр {name} должен быть непустой строкой."}
        if "part_id" in args and not re.fullmatch(PART_PATTERN, args["part_id"]):
            return {"error": "Номер детали должен иметь вид P-1042 или P-1042B."}
        if "warehouse" in args and args["warehouse"] not in WAREHOUSES:
            return {"error": "Неизвестный склад. Допустимые склады: " + ", ".join(WAREHOUSES)}
        if "qty" in args:
            qty = args["qty"]
            if type(qty) is not int or qty <= 0:
                return {"error": "Количество должно быть целым числом больше нуля."}

        try:
            return function(**args)
        except erp_client.ErpNotFound as exc:
            return {"error": f"В ERP нет данных: {exc}"}
        except erp_client.ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    return call


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    # TODO: пять функций поверх local_search, global_search и клиента ERP
    def search_knowledge(question: str) -> dict:
        result = local_search(question, kb, llm)
        return {"answer": result.text, "facts": result.facts, "sources": result.sources, "refused": result.refused}

    def summarize_topic(question: str) -> dict:
        result = global_search(question, kb, llm)
        return {"answer": result.text, "sources": result.sources, "refused": result.refused}

    functions = {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }
    tools = {}
    for schema in TOOL_SCHEMAS:
        spec = schema["function"]
        name = spec["name"]
        properties = spec["parameters"]["properties"]
        tools[name] = _with_validation(functions[name], properties)
    return tools
