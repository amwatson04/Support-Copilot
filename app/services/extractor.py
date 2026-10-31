"""Извлечение фактов из текста игрока без LLM: ссылки на правила, ник, доказательства.

Всё, что найдено здесь, — заявление игрока, а не доказанный факт.
"""
import re

from app.services.rules import ALIASES, UNKNOWN_CODES, filter_existing, make_id

# Номер пункта: 2.1, 1.1.2, 5.4.1. Не цепляем куски дат и сумм вроде 10.10.2026
_NUMBER = r"(?<![\d.])(\d{1,2}(?:\.\d{1,2}){1,2})(?![\d])"
# Слово сразу после номера: «2.1 ОП», «2.1оп», «1.2 GZ»
_REF_AFTER = re.compile(_NUMBER + r"\s*([A-Za-zА-Яа-яЁё]+)?")
# Код перед номером: «ОП 2.1», «GZ 1.2»
_REF_BEFORE = re.compile(r"(?<![A-Za-zА-Яа-яЁё])([A-Za-zА-Яа-яЁё]{2,6})\s*" + _NUMBER)

# Синтетические ники в датасете вида Имя_Фамилия или Имя_Фамилия12
_NICK = re.compile(r"\b([A-Z][a-z]+_[A-Z][a-z]+\d*)\b")
# Номер сервера рядом со словом «сервер»: «сервера номер 18», «25 сервера», «сервера 07», «server 3».
# (?!\.?\d) — чтобы не принять пункт правил «2.1» за сервер 2 (а точка в конце предложения допустима)
_SERVER_AFTER = re.compile(r"(?:сервер|server)\w*\s*(?:номер|№|#)?\s*(\d{1,2})(?!\.?\d)", re.IGNORECASE)
# (?<![\w.]) — цифры не должны быть частью ника: «Test_Player43 Server» — это не сервер 43
_SERVER_BEFORE = re.compile(r"(?<![\w.])(\d{1,2})\s*(?:-?(?:й|ой|ом|го|м))?\s*(?:сервер|server)", re.IGNORECASE)
# Ссылка на видео тоже доказательство (в датасете ссылки заменены на <url>)
_LINK = re.compile(r"<url>|https?://", re.IGNORECASE)


def _prefix_for(code: str | None) -> str | None:
    """Префикс документа по коду рядом с номером. None = ссылку отбросить."""
    if not code:
        return ""  # голый номер считаем общими правилами
    code = code.upper()
    if code in ALIASES:
        return ALIASES[code]
    if code in UNKNOWN_CODES:
        return None
    return ""  # обычное слово после номера («2.1 и оскорбления») — тоже общие правила


def extract_rule_refs(text: str) -> list[str]:
    """Возвращает ID правил, которые упомянул игрок и которые реально существуют."""
    found: list[tuple[int, str]] = []
    taken: set[int] = set()  # позиции номеров, для которых код уже найден перед номером

    for m in _REF_BEFORE.finditer(text):
        code = m.group(1).upper()
        if code in ALIASES:
            found.append((m.start(2), make_id(ALIASES[code], m.group(2))))
            taken.add(m.start(2))

    for m in _REF_AFTER.finditer(text):
        if m.start(1) in taken:
            continue
        prefix = _prefix_for(m.group(2))
        if prefix is not None:
            found.append((m.start(1), make_id(prefix, m.group(1))))

    found.sort()
    return filter_existing([rule_id for _, rule_id in found])


def extract_nickname(text: str) -> str | None:
    match = _NICK.search(text)
    return match.group(1) if match else None


def extract_server(text: str) -> str | None:
    """Номер сервера, если игрок его назвал. «07» и «7» — один сервер, возвращаем «7»."""
    match = _SERVER_AFTER.search(text) or _SERVER_BEFORE.search(text)
    return str(int(match.group(1))) if match else None


def has_evidence(text: str, attachments: int) -> bool:
    """Есть ли что-то, похожее на доказательство. Содержимое скриншотов нам неизвестно."""
    return attachments > 0 or bool(_LINK.search(text))
