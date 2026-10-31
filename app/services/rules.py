"""Реестр пунктов правил из data/rules.json.

Проблема: номер «1.1» есть в 10 разных документах правил.
Поэтому у каждого документа свой префикс, а ID пункта = префикс + номер:
    "2.1"        — общие правила (без префикса, как в контракте API)
    "GZ 1.2"     — правила зелёной зоны
    "ЖБ 1.1.2"   — правила жалоб
Только ID из этого реестра можно цитировать в ответе.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "rules.json"

# id документа в rules.json -> префикс для цитирования
PREFIX_BY_DOC = {
    30: "",      # Общие правила игры (ОП)
    31: "ЖБ",    # Правила жалоб
    32: "GZ",    # Правила нахождения в Зелёной Зоне
    33: "ОПЛЗ",  # Правила для лидеров и их заместителей
    34: "ПГО",   # Правила государственных организаций
    35: "ПГВД",  # Правила государственной волны и департамента
    36: "FW",    # Правила семейных мероприятий
    40: "ОПГ",   # Правила для ОПГ
    41: "ВБ",    # Правила войны за бизнес
    42: "БЗФ",   # Правила битвы за фургон
}

# Как игроки и админы пишут названия документов -> наш префикс.
# Взято из частот в tickets_train: «2.1 ОП», «2.1 OP», «1.2 ГЗ», «1.1 ППЖ» и т.д.
ALIASES = {
    "ОП": "", "OP": "", "OП": "", "ОР": "", "ОБЩИХ": "",
    "ЖБ": "ЖБ", "ППЖ": "ЖБ", "ПЖ": "ЖБ",
    "GZ": "GZ", "ГЗ": "GZ", "ЗЗ": "GZ", "ZZ": "GZ", "GREEN": "GZ", "GREENZ": "GZ",
    "ОПЛЗ": "ОПЛЗ",
    "ПГО": "ПГО",
    "ПГВД": "ПГВД",
    "FW": "FW", "ФВ": "FW",
    "ОПГ": "ОПГ",
    "ВБ": "ВБ",
    "БЗФ": "БЗФ", "FURA": "БЗФ",
}

# Коды документов, которых нет в rules.json (УК, ФП и т.п.).
# Ссылки с ними отбрасываем, а не приписываем к общим правилам.
UNKNOWN_CODES = {"УК", "ФП", "ПН", "ППО", "ППИО", "GW"}


@dataclass(frozen=True)
class RulePoint:
    id: str         # ID для цитирования, например "GZ 1.2"
    document: str   # название документа правил
    text: str


def make_id(prefix: str, ref: str) -> str:
    return f"{prefix} {ref}" if prefix else ref


@lru_cache(maxsize=1)
def load_rules() -> dict[str, RulePoint]:
    """Читает rules.json один раз и возвращает словарь ID -> пункт."""
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    registry: dict[str, RulePoint] = {}
    for doc in data["rules"]:
        prefix = PREFIX_BY_DOC.get(doc["id"])
        if prefix is None:
            continue  # документ без пунктов (RP термины) или новый, неизвестный нам
        for point in doc["points"]:
            rule_id = make_id(prefix, point["ref"])
            registry[rule_id] = RulePoint(id=rule_id, document=doc["category"], text=point["text"])
    return registry


def rule_exists(rule_id: str) -> bool:
    return rule_id in load_rules()


def get_rule(rule_id: str) -> RulePoint | None:
    return load_rules().get(rule_id)


def filter_existing(rule_ids: list[str]) -> list[str]:
    """Оставляет только существующие ID, без повторов, в исходном порядке."""
    return [r for r in dict.fromkeys(rule_ids) if rule_exists(r)]
