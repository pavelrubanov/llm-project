"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, argument_error
from assistant.infra import llm as llm_api
from assistant.infra import schemas
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import AgentResult

_INSTRUCTION = (
    "Ты помогаешь диспетчеру холодильного оборудования. Отвечай кратко по результатам инструментов. "
    "Для конкретной модели, ошибки или детали вызови search_knowledge, для обзора — summarize_topic. "
    "В аргумент question скопируй вопрос диспетчера целиком, слово в слово. Учитывай последнюю замену детали. "
    "Цену запрашивай через get_price, остаток — через check_stock. Если нужны оба, проверь оба. "
    "«Какую деталь заказать?» — вопрос о детали. По прямой просьбе оформить заказ вызови create_service_order. "
    "Если количество или склад неизвестны, уточни их. Если сведений нет, напиши «В базе знаний не найдено.»"
)
_NO_DATA = "В базе знаний не найдено."
_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\d-])"
_AMOUNT = re.compile(rf"({_NUMBER})\s*(руб|₽|шт)", re.IGNORECASE)
_LABELED_AMOUNT = re.compile(rf"\b(цена|стоимость|стоит|остаток|осталось)\s*[:—-]?\s*({_NUMBER})", re.IGNORECASE)
_SMALL = (
    "ноль один два три четыре пять шесть семь восемь девять десять "
    "одиннадцать двенадцать тринадцать четырнадцать пятнадцать "
    "шестнадцать семнадцать восемнадцать девятнадцать"
)
_TENS = "двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто"
_HUNDREDS = "сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот"
_COUNT_WORDS = {word: i for i, word in enumerate(_SMALL.split())}
_COUNT_WORDS.update({word: 10 * i for i, word in enumerate(_TENS.split(), 2)})
_COUNT_WORDS.update({word: 100 * i for i, word in enumerate(_HUNDREDS.split(), 1)})
_COUNT_WORDS.update({"одну": 1, "одна": 1, "одно": 1, "две": 2})
_WORD = "|".join(_COUNT_WORDS)
_COUNT = rf"(?<![\w.,–—-])(?:\d+(?:[ \u00a0\u202f]\d{{3}})*|(?:{_WORD})(?:\s+(?:{_WORD}))*)\b"
_CITIES = {
    "москв": "Москва",
    "казан": "Казань",
    "санкт-петербург": "Санкт-Петербург",
    "новосибирск": "Новосибирск",
}


def _warehouses(text: str) -> set[str]:
    return {city for stem, city in _CITIES.items() if re.search(rf"\b{stem}[а-я]*\b", text.casefold())}


def _quantity_in_question(qty: int, question: str) -> bool:
    text = question.casefold()
    text = re.sub(r"\b\d+\s*[-–—.,]\s*\d+\b", "?", text)
    command = r"\b(?:заказ(?:ать)?|закажи(?:те)?|количество|кол-во|нужно|нужны)"
    after_command = rf"{command}[\s:—-]+(?:на\s+)?({_COUNT})(?!\s*(?:руб|₽|тысяч|миллион|%))"
    if re.search(rf"({_COUNT})\s+(?:или|до|тысяч[а-я]*|миллион[а-я]*)\b", text):
        return False
    matches = re.findall(after_command, text)
    if not matches:
        matches = re.findall(rf"({_COUNT})\s*(?:шт|единиц|детал|экземпляр)", text)
    values = set()
    for raw in matches:
        digits = "".join(raw.split())
        if digits.isdigit():
            values.add(int(digits))
        else:
            values.add(sum(_COUNT_WORDS[word] for word in raw.split()))
    return values == {qty}


def _observation(data: dict) -> str:
    if "answer" in data:
        return data["answer"]
    if "price_rub" in data:
        return f'{data["part_id"]} стоит {data["price_rub"]} руб.'
    if "qty" in data:
        return f'{data["part_id"]}, склад {data["warehouse"]}: {data["qty"]} шт.'
    return ""


def _amounts(text: str) -> set[tuple[str, str]]:
    amounts = set()
    for raw, unit in _AMOUNT.findall(text):
        kind = "шт" if unit.casefold() == "шт" else "руб"
        amounts.add(("".join(raw.split()), kind))
    for label, raw in _LABELED_AMOUNT.findall(text):
        kind = "шт" if label.casefold() in {"остаток", "осталось"} else "руб"
        amounts.add(("".join(raw.split()), kind))
    return amounts


def _numbers(text: str) -> set[str]:
    return {"".join(raw.split()) for raw in re.findall(_NUMBER, text)}


def _erp_amounts_match(answer: str, steps: list[schemas.ToolStep]) -> bool:
    readings = [step.result for step in steps if step.result]
    separator = r"[;\n]|[.!?](?:\s|$)|,\s*(?=(?:а |и )?(?:в |на складе |p-))"
    for clause in re.split(separator, answer.casefold().replace("*", "")):
        words = rf"\b(?:{_WORD}|тысяч[а-я]*)\b"
        units = r"(?:руб|₽|шт|штук|единиц)"
        if re.search(rf"{words}\s*{units}|\b(?:цена|стоимость|остаток|осталось|стоит)[\s:—-]+{words}", clause):
            return False
        parts = {code for code in find_codes(clause) if code.startswith("P-")}
        cities = _warehouses(clause)
        amounts = _amounts(clause)
        if cities and not amounts:
            amounts = {(value, "шт") for value in _numbers(clause)}
        for value, unit in amounts:
            field = "qty" if unit == "шт" else "price_rub"
            candidates = [row for row in readings if field in row]
            if parts:
                candidates = [row for row in candidates if row["part_id"] in parts]
                if {row["part_id"] for row in candidates} != parts:
                    return False
            if cities and field == "qty":
                candidates = [row for row in candidates if row["warehouse"] in cities]
                if {row["warehouse"] for row in candidates} != cities:
                    return False
            values = {str(row[field]) for row in candidates}
            if values != {value}:
                return False
    return True


def _answer_error(answer: str, known_codes: set[str], evidence: str, steps: list[schemas.ToolStep]) -> str | None:
    if not steps and "?" not in answer:
        return "Сначала получи данные через подходящий инструмент."
    order_status = r"\b(?:оформлен|создан|подтвержд[её]н|зарегистрирован|заказан)\b|\bSO-\d+\b"
    if re.search(order_status, answer, re.IGNORECASE):
        return "Создание заказа подтверждает только ERP. Для оформления вызови create_service_order."
    if set(find_codes(answer)) - known_codes or _numbers(answer) - _numbers(evidence):
        return "Исправь ответ. Возьми коды и числа только из вопроса и результатов инструментов."
    if not _erp_amounts_match(answer, steps):
        return "Запиши значения ERP цифрами. Для каждой детали и склада укажи цену или остаток отдельной строкой."
    return None


def _call_error(call: llm_api.ToolCall, tools: dict, known_codes: set[str]) -> str | None:
    if call.name not in tools:
        return "Неизвестный инструмент."
    args = call.arguments
    error = argument_error(call.name, args)
    if not isinstance(args, dict):
        return error
    part = args.get("part_id")
    if isinstance(part, str) and part not in known_codes:
        return "Найди артикул через search_knowledge: спроси, какая деталь нужна для этой модели и узла или ошибки."
    if error:
        return error
    search_codes = find_codes(args.get("question", ""))
    if any(code.startswith("P-") and code not in known_codes for code in search_codes):
        return "Используй исходный вопрос диспетчера без новых номеров деталей."
    if call.name == "create_service_order":
        models = {code for code in known_codes if code_type(code) == "Model"}
        if args["model"] not in models:
            return "Уточни модель оборудования у диспетчера."
    return None


def _execute(function: Callable, args: dict) -> dict:
    for _ in range(3):
        try:
            return function(**args)
        except ErpUnavailable:
            continue
    raise ErpUnavailable("Данные ERP получить не удалось.")


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    messages = [{"role": "system", "content": _INSTRUCTION}, {"role": "user", "content": question}]
    result = AgentResult(_NO_DATA)
    known_codes = set(find_codes(question))
    requested_models = {code for code in known_codes if code_type(code) == "Model"}
    previous = {}
    observations = []
    no_data = False
    corrected = False

    while len(result.steps) < max_steps:
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            answer = reply.content.strip()
            evidence = "\n".join([question, *observations])
            error = _answer_error(answer, known_codes, evidence, result.steps)
            if error and not corrected:
                messages.append({"role": "user", "content": error})
                corrected = True
                continue
            refusal = answer.casefold().startswith("в базе знаний не найдено")
            if error or not answer or refusal or (no_data and not observations):
                result.refused = True
            else:
                result.answer = answer
            return result

        for call in reply.tool_calls:
            if len(result.steps) >= max_steps:
                break
            name, args = call.name, call.arguments
            step = schemas.ToolStep(name, args)
            result.steps.append(step)
            error = _call_error(call, tools, known_codes)
            if not error and name == "create_service_order":
                if requested_models and requested_models != {args["model"]}:
                    step.error = "Модель не подтверждена вопросом."
                    result.answer = "Для какой модели оформить заказ?"
                    return result
                if not _quantity_in_question(args["qty"], question):
                    step.error = "Количество не подтверждено вопросом."
                    result.answer = "Сколько штук заказать?"
                    return result
                if _warehouses(question) != {args["warehouse"]}:
                    step.error = "Склад не подтверждён вопросом."
                    result.answer = "На каком складе оформить заказ?"
                    return result
                if not confirm:
                    step.error = "Ожидается подтверждение."
                    result.needs_confirmation = True
                    result.pending_action = {"tool": name, "args": args}
                    result.answer = (
                        f'Подтвердите заказ {args["part_id"]} для {args["model"]}. '
                        f'Склад — {args["warehouse"]}, количество — {args["qty"]}.'
                    )
                    return result

            key = (name, json.dumps(args, sort_keys=True))
            if error:
                data = {"error": error}
            elif key in previous:
                data = previous[key]
            else:
                try:
                    data = _execute(tools[name], args)
                except ErpUnavailable as exc:
                    step.error = str(exc)
                    result.answer = str(exc)
                    return result
                previous[key] = data
            messages.append(llm_api.tool_message(call, data))
            step.error = data.get("error")
            if step.error:
                if step.error.startswith("В ERP нет данных"):
                    no_data = True
                continue
            step.result = data
            if data.get("refused"):
                no_data = True
                continue
            if "order_id" in data:
                result.answer = f'Заказ {data["order_id"]} создан.'
                return result
            known_codes.update(find_codes(json.dumps(data, ensure_ascii=False)))
            result.sources = sorted(set(result.sources).union(data.get("sources", [])))
            observation = _observation(data)
            if observation and observation not in observations:
                observations.append(observation)

    if no_data and not observations:
        result.refused = True
        observations.append(_NO_DATA)
    observations.append("Достигнут лимит шагов, работа прервана.")
    result.answer = " ".join(observations)
    return result
