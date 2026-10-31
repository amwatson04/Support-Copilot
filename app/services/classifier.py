"""Классификация по ключевым словам — без LLM.

Зачем, если есть LLM:
1. Самые частые обращения короткие и однозначные: «оск», «дм», «опру», «помогите».
   Их не нужно отправлять в LLM — экономим квоту бесплатного тарифа (~15 запросов/мин).
2. Когда LLM недоступна (таймаут, 429, LLM_PROVIDER=disabled), лучше осмысленная категория
   по словам, чем «complaint.other» для всего подряд.

Маркеры взяты из самых частых текстов в tickets_train (сленг, транслит, узбекский).
"""
import re
from dataclasses import dataclass

from app.services.escalation import SENSITIVE_PATTERNS

_F = re.IGNORECASE | re.UNICODE

# Жалобы: категория по пункту правил, который указал игрок
_REF_TO_CATEGORY = {
    "2.1": "complaint.insult", "2.4": "complaint.insult", "2.6": "complaint.insult",
    "5.2": "complaint.cheat",
    "5.3": "complaint.dm", "5.4": "complaint.dm",
    "4.1": "complaint.scam", "4.2": "complaint.scam",
    "2.3": "complaint.rmt_ads", "4.4": "complaint.rmt_ads",
}

# Порядок важен: сначала более специфичные категории
_COMPLAINT_KEYWORDS = [
    ("complaint.cheat", re.compile(r"чит|софт|soft|cheat|аим|aim\b|\bwh\b|вх\b|летает|fly", _F)),
    ("complaint.scam", re.compile(r"скам|scam|обман|кинул|развел|развёл|aldad|aldab", _F)),
    ("complaint.rmt_ads", re.compile(r"реклам|прода\w*\s+(акк|аккаунт|вирт).{0,30}(реал|тг|руб|тенге)|rmt", _F)),
    ("complaint.insult", re.compile(
        r"\bоск|оскор|оскар|обзыв|матер|мат(ом|ы)\b|упом\w*\s+род|\bрод(ни|ных|ня)\b|\bosk|onadan|\bso'?k|"
        r"haqorat|chestdan|балағат|balağat|боқта|insult|xing|"  # kk-латиница, kk, pt — не проверены на данных
        # uz по пропускам в train: «haromi», «sukindi», «soʻkib» (узбекский апостроф ʻ); uz/kk кириллица — те же слова
        r"\bharomi?\b|\bso[ʻ‘’`]k|\bsu[ʻ']?k(in|ib|di)|ҳақорат|хақорат|\bсўк|\bсук(ди|иб|ин)|сөкт|боктад", _F)),
    ("complaint.dm", re.compile(
        r"\bдм\b|\bdm\b|\bдб\b|\bdb\b|\bтк\b|\btk\b|\bск\b|убил|убийств|убива|\bзз\b|\bgz\b|\bгз\b|связк|"
        r"повязал|o'?ldir|uldir|nonrp|нон\s*рп|без\s*причин|өлтір|öltir|"
        # kk/uz «убил / застрелил без причины» (кириллица и латиница); апелляции «без причины бан» проверяются раньше
        r"ўлдир|улдир|отиб\s+ташла|otib\s+tashla|атып\s+таста|atyp\s+tasta|sababsiz|сабабсиз|себепсіз|sebepsı?z", _F)),
]

_REPORT_RULES = [
    ("report.account", "p0", re.compile(r"взлом|hack|пропал\w*\s+(машин|авто|дом|деньг|имущ|вещ)|акк\w*\s+украл", _F)),
    ("report.help", "p2", re.compile(
        r"застрял|текстур|переверн|перевер|наручник|спавн|spawn|выта|почин|"
        r"stuck|zastr|tiqil|tekstur", _F)),
    ("report.complaint", "p2", re.compile(
        r"\bдм\b|\bdm\b|чит|софт|\bоск|оскор|убива|нон\s*рп|nonrp|\bтк\b|заломал|мешает|"
        r"объясн|обьясн|за\s*что|наказ|\bмут|варн|посадил|пересмотр", _F)),
    # Просьба телепортировать — не помощь, а оффтоп: администрация такую услугу не оказывает (решение команды)
    ("report.offtopic", "p3", re.compile(r"телепорт|\bтп\b|\btp\b", _F)),
    ("report.offtopic", "p3", re.compile(
        r"^\W*(помоги\w*|помогі\w*|помагите|памаги\w*|помгите|помощь|спасите|хелп|help(\s+me)?|sos|сос|ау+|"
        r"привет\w*|админ\w*|yordam|\+)\W*$", _F)),
    ("report.question", "p3", re.compile(r"\?|^\s*(как|где|когда|можно|почему|сколько)\b", _F)),
]

_PRIORITY = {
    "complaint.insult": "p2", "complaint.dm": "p2", "complaint.admin": "p2", "complaint.other": "p2",
    "complaint.cheat": "p1", "complaint.scam": "p1", "complaint.rmt_ads": "p1",
    "complaint.invalid": "p3",
    "report.account": "p0", "report.payment": "p0", "report.help": "p2", "report.complaint": "p2",
    "report.question": "p3", "report.offtopic": "p3",
}


def category_priority(category: str, text: str) -> str:
    """Приоритет по правилам таксономии для категории (как в классификации по словам).
    Используется и для ответа LLM: замер 2026-10-31 — qwen2.5:3b ставила p1 оскорблениям и DM (эталон: p2)."""
    if category == "report.complaint" and re.search(r"чит|софт|cheat|soft", text, _F):
        return "p1"
    return _PRIORITY.get(category, "p2")

# Короткий текст без лишних слов можно уверенно классифицировать без LLM
_SHORT_TEXT_CHARS = 25
_NOISE = re.compile(r"<\w+>|\d+(?:\.\d+)*\s*(оп|op|gz|гз|фв|fw)?|[^\w\s]", _F)


@dataclass
class KeywordResult:
    category: str
    priority: str
    confident: bool  # True — можно не звать LLM


def classify_by_keywords(kind: str, text: str, player_refs: list[str]) -> KeywordResult | None:
    """Возвращает категорию по словам или None, если ничего не подошло."""
    core = " ".join(_NOISE.sub(" ", text).split())
    short = len(core) <= _SHORT_TEXT_CHARS

    if kind == "report":
        for category, priority, pattern in _REPORT_RULES:
            if pattern.search(text):
                if category == "report.complaint" and re.search(r"чит|софт", text, _F):
                    priority = "p1"
                return KeywordResult(category, priority, confident=short)
        return None

    # Апелляция на своё наказание — жалоба на действия администратора
    if SENSITIVE_PATTERNS["appeal"][1].search(text):
        return KeywordResult("complaint.admin", "p2", confident=short)
    for ref in player_refs:
        category = "complaint.dm" if ref.startswith("GZ ") else _REF_TO_CATEGORY.get(ref)
        if category:
            return KeywordResult(category, _PRIORITY[category], confident=short)
    for category, pattern in _COMPLAINT_KEYWORDS:
        if pattern.search(text):
            return KeywordResult(category, _PRIORITY[category], confident=short)
    return None
