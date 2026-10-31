"""Когда нужен человек, и проверка черновика на запрещённые обещания.

Всё здесь — правила в коде, а не решение LLM: так их нельзя «уговорить»
текстом обращения, и их легко объяснить жюри.
Слова-маркеры взяты из реальных обращений в tickets_train.
"""
import re
from functools import lru_cache

from app.services.rules import load_rules

_FLAGS = re.IGNORECASE | re.UNICODE

# Обращение игрока: что всегда передаём человеку
SENSITIVE_PATTERNS = {
    "appeal": (
        "Апелляция на наказание — решает администратор",
        re.compile(
            r"не\s*согл|несогл|опру|орпу|опровер|оспор|обжалов|пересмотр|сним\w*\s+(наказ|бан|мут|варн)|"
            r"сня\w*\s+(наказ|бан|мут|варн)|разбан|unban|appeal|требую\s+док|(за\s*что|зашто|за\s+шо)\s+(бан|мут|варн)|"
            # слепая оценка этапа 2 (2026-10-31): апелляция вида «… заблокирован администратором …»
            r"заблокирова\w*\s+(администратор|админ)|"
            r"ne\s*sogl|bandan\s+chiqar|"
            # Добавлено 2026-10-31 по пропущенным апелляциям в tickets_train (подбор на train, не на эталоне):
            # ru — «дал варн без причины», «не правильно выдал», «предоставьте доказательства», «я не нарушал»
            r"(дал|дали|дала|выдал\w*|влепил\w*)\s+(мне\s+|нам\s+)?(варн|мут|бан|наказан\w*|деморган\w*|джайл|jail|warn)"
            r".{0,40}(без\s*причин|просто\s*так|ни\s*за\s*что|не\s*за\s*что|незачто|зря|ошибочн|неправильн|не\s*правильн)|"
            r"(без\s*причин\w*|просто\s*так|ни\s*за\s*что|незачто)\s.{0,25}(дал|дали|выдал\w*|забанил\w*|замутил\w*)|"
            r"(не\s*правильно|неправильно|ошибочно|неверно|зря)\s+(выдал\w*|выдан\w*|дал\w*|наказ\w*|забанил\w*)|"
            r"меня\s+(забанил\w*|замутил\w*|наказал\w*|кикнул\w*)|предоставьте\s+(доказательств|док)\w*|"
            r"я\s+не\s+(нарушал|убивал|оскорблял|оскал|использовал|читер|багоюз)|"
            # uz — прошедшее время «мне дали» (berdi/berishdi), а не просьба «дайте ему» (bering)
            r"(jazo|ban|warn|mute|blok)\w*\s+(berdi|berishdi|bervot|beribdi|berdingiz|berishyapti)|"
            r"bandan\s+yech|rozi\s+emas|xato\s+jazo|nohaq|jazo\w*\s+(xato|noto'?g'?ri)|demorganga\s+tiq|"
            r"ishlatmaganman|qilmaganman|(soft|chit|cheat)\w*\s+emasman|"
            # Добавлено 2026-10-31 по пропущенным uz-апелляциям в train: «без причины» + наказание, «noroziman». «Без причины» — апелляция только рядом со словом о наказании (иначе это DM: «sababsiz otib»)
            r"(sababsiz|be\s*sabab|bez\s*prichin\w*|nima\s*sabab\w*|сабабсиз|ноҳақ|себепсіз|sebepsı?z)\W+(\w+\W+){0,3}"
            r"(?<!\w)(ban|бан|mute?|мут|warn|варн|jazo|жазо|жаза|jaza|qama\w*|қама\w*|blok\w*|блок\w*|berildi|berdi\w*)"
            r"(?![кk])|"
            # (?<!\w)…(?![кk]) — «бан», но не «банк»: «bank oldida sababsiz otib» — это DM, а не апелляция
            r"(?<!\w)(ban|бан|mute?|мут|warn|jazo|жазо|жаза|qama\w*)(?![кk])\w*\s+(\w+\s+){0,2}"
            r"(sababsiz|be\s*sabab|сабабсиз|себепсіз)|"
            r"\bnorozi(man)?\b|норози(ман)?|cheklov\w*\s+olib\s+tashla|келіспеймін|kelıspeimın|kelispeymin|"
            # kk/uz кириллица — «мне дали бан»: бан берді / бан беришди
            r"(?<!\w)(бан|мут|жаза|жазо|варн)(?!к)\w*\s+(берді|беришди|берди|салды)",
            _FLAGS,
        ),
    ),
    "hacking": (
        "Подозрение на взлом аккаунта — решает администратор",
        re.compile(r"взлом|hack|украл\w*\s+акк|parol\w*\s+o'?zgar|akkaunt\w*\s+o'?g'?irla", _FLAGS),
    ),
    "payment": (
        "Вопрос оплаты или доната — решает администратор",
        re.compile(
            r"оплат|платеж|платёж|списал|возврат\w*\s+(денег|средств)|"
            r"донат\w*.{0,30}(не\s*приш|не\s*начисл|не\s*получ)|(не\s*приш|не\s*начисл)\w*.{0,30}(донат|гк|деньг)|"
            r"payment|refund|paid|to'?lov|reembols|pagamento|pago",
            _FLAGS,
        ),
    ),
    "threat": (
        "Угрозы вне игры — решает администратор",
        re.compile(r"угрож\w*.{0,20}(убить|найти|приеду|деанон)|деанон|swat|адрес\w*\s+(найд|узна)", _FLAGS),
    ),
}

# Черновик ответа: чего мы никогда не обещаем и не утверждаем
FORBIDDEN_IN_REPLY = [
    ("Обещание разбана/снятия наказания", re.compile(
        r"разбан|сним\w*\s+(вам\s+)?(наказ|бан|блок)|будет\s+снят|unban|lift\w*\s+(the\s+)?ban|bandan\s+chiqar|"
        r"desban|desbloque|bandan\s+olib", _FLAGS)),
    ("Обещание возврата денег/имущества", re.compile(
        r"верн(ём|ем|ут|ется|ётся)\s+.{0,20}(деньг|средств|имуществ|машин|донат)|компенсац|refund|reimburs|"
        r"compensat|qaytar|reembols|devolv|қайтар|"
        # слепая оценка 2026-10-31: «Ваш аккаунт будет восстановлен»
        r"(аккаунт\w*|имуществ\w*|машин\w*|доступ\w*)\s+(будет|будут)\s+(восстановлен|возвращ)\w*|восстановим|"
        r"account\s+will\s+be\s+(restored|recovered)", _FLAGS)),
    ("Обещание наказать другого игрока", re.compile(
        r"(будет|получит)\s+(наказан|заблокирован|забанен|наказание)|накажем|забаним|"
        r"(будет|будут)\s+принят\w*.{0,30}наказан|"
        r"will\s+be\s+(banned|punished|blocked)|jazolan|жазалан|será\s+(sancionad|banead|castigad|punid|banid)", _FLAGS)),
    ("Утверждение о содержимом скриншотов", re.compile(
        r"(на\s+)?скрин\w*\s+(видно|подтвержд|доказ)|screenshots?\s+(show|prove|confirm)|"
        r"skrinshot\w*\s+(ko'?rin|isbot)", _FLAGS)),
    # Найдено в реальном тесте Ollama: «Ваша жалоба на оскорбления подтверждается»
    ("Утверждение, что нарушение подтверждено", re.compile(
        r"(жалоб|нарушени)\w*.{0,40}(подтвержда|подтвержден|доказан)|подтвержда\w*\s+нарушени|"
        # повторный замер Ollama 2026-10-31: «Ваше сообщение подтверждает использование читов»
        r"подтвержда\w*\s+(факт\w*\s+)?(использовани|чит|софт|оскорблени|убийств)|"
        r"(the|this|your)\s+violation\s+(is|was|has\s+been)\s+confirmed|confirmed\s+(the\s+)?violation|tasdiqlan|"
        # слепая оценка 2026-10-31: «This behavior is clearly a cheat»
        r"(явно|очевидно|точно)\s+(чит|нарушени|использ)|clearly\s+(a\s+)?(cheat|violation|hack)|"
        r"(is|was)\s+(definitely|clearly|obviously)\s+(a\s+)?(cheat|violat)", _FLAGS)),
    # Повторный замер Ollama 2026-10-31: «Сожалеем, что вы получили наказание по необоснованным причинам» —
    # решение по апелляции принимает администратор, черновик не может признать наказание ошибочным
    # слепая оценка этапа 2 (Ollama): «Жалоба будет отклонена без рассмотрения», «Максимальная санкция — блокировка
    # на 10 дней» — исход и размер наказания решает администратор, не черновик
    ("Решение по обращению или размер наказания", re.compile(
        r"(жалоб|обращени|заявлени|апелляци)\w*\s+(будет\s+|была\s+)?(отклонен|отказан|удовлетворен|одобрен)|"
        r"максимальн\w*\s+(санкци|наказани)|(получит|дадут|выдадут)\s+(бан|блокировк|мут)\w*\s+на\s+\d", _FLAGS)),
    ("Утверждение, что наказание ошибочно", re.compile(
        r"(наказани|бан|мут|варн|блокировк)\w*.{0,40}(необоснованн|несправедлив|ошибочн|по\s+ошибке|незаслуженн)|"
        r"(необоснованн|несправедлив|ошибочн|незаслуженн)\w*\s+(наказ|бан|мут|варн|блок)|"
        r"(punishment|ban|mute)\s+(was|is)\s+(unfair|unjustified|a\s+mistake|wrong)|nohaq\s+(jazo|ban)", _FLAGS)),
    # слепая оценка 2026-10-31: «Ваш платеж был успешно обработан» — модель не знает статуса платежа
    ("Утверждение о статусе платежа", re.compile(
        r"плат[её]ж\w*\s+(был\w*\s+)?(успешно\s+)?(обработан|зачислен|получен|прош[её]л|проведён|проведен)|"
        r"(деньги|средства|гк|донат)\s+(уже\s+)?(зачислен|поступил|начислен)|"
        r"payment\s+(was|has\s+been)\s+(successfully\s+)?(processed|received|credited)", _FLAGS)),
]
# Для утверждений о нарушении условная форма допустима: «примем меры, ЕСЛИ нарушение подтвердится»
_CONDITIONAL_ONLY = {"Утверждение, что нарушение подтверждено", "Утверждение, что наказание ошибочно"}
_CONDITIONAL = re.compile(r"\b(если|if|agar|егер|si|se)\b[^.!?]{0,40}$", _FLAGS)

# Найдено в реальном тесте Ollama: выдуманный адрес «support@grandmobile.com».
# Контакты и ссылки в черновике разрешены, только если они есть в официальных правилах (rules.json):
# выдуманному контакту игрок поверит, а официальный (например, техподдержка из правил жалоб) полезен.
_CONTACT = re.compile(
    r"https?://[^\s,;)»\"']+|www\.[^\s,;)»\"']+|t\.me/[^\s,;)»\"']+|discord\.gg/[^\s,;)»\"']+|"
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}|\+?\d[\d\s()-]{8,}\d", _FLAGS)
UNOFFICIAL_CONTACT = "Контакт или ссылка, которых нет в официальных правилах (может быть выдуманной)"


def _normalize_contact(value: str) -> str:
    return value.lower().rstrip(".,!?:/")


@lru_cache(maxsize=1)
def _official_contacts() -> frozenset[str]:
    return frozenset(_normalize_contact(m.group(0)) for rule in load_rules().values()
                     for m in _CONTACT.finditer(rule.text))


# Категории, где наказание или действие в игре выполняет только администратор
_ADMIN_ACTION_CATEGORIES = {
    "complaint.insult", "complaint.cheat", "complaint.dm", "complaint.scam",
    "complaint.rmt_ads", "complaint.admin", "complaint.other",
    "report.help", "report.complaint", "report.account", "report.payment",
}
LOW_CONFIDENCE = 0.5


# Адресат черновика — тот, кто написал обращение. Реальные ошибки (Gemini и Ollama, 2026-10-31):
# заявителю жалобы на оскорбления писали «воздержитесь от ненормативной лексики»,
# игроку с апелляцией — «ваше поведение нарушает правила». Модель перепутала адресата
# или объявила виновным того, чья вина не установлена.
_TELL_READER_TO_STOP = re.compile(
    r"\bне\s+(используйте|оскорбляйте|нарушайте|убивайте|материтесь|обманывайте|выражайтесь)|"
    r"воздерж\w+|прекратите|извинитесь|будьте\s+(вежлив|корректн)|"
    r"refrain|stop\s+(using|insulting|cheating|killing)|(do\s+not|don'?t)\s+(use|insult|cheat|kill|swear)|"
    r"apologi[sz]e|"
    r"ishlatmang|haqorat\s+qilmang|so'?kinmang|to'?xtating|kechirim\s+so'?rang|"
    r"қолданбаңыз|балағаттамаңыз|тоқтатыңыз|кешірім\s+сұраңыз|"
    r"absténgase|deje\s+de|no\s+(use|insulte)|abstenha-se|pare\s+de|não\s+(use|insulte)", _FLAGS)
_READER_IS_GUILTY = re.compile(
    r"ваш\w*\s+(поведение|сообщени\w+|действи\w+|нарушени\w+)\s+(наруша|содерж|противореч|недопуст)|"
    # слепая оценка этапа 2 (Ollama): «Ваше поведение в чате нарушает правила», «Ваш никнейм не соответствует правилам»
    r"ваш\w*\s+(поведение|никнейм|ник)\b.{0,30}(наруша|противореч|недопуст|не\s+соответству)|"
    r"соблюдайте\s+(правил|этичн)|используйте\s+разрешенн|"
    r"вы\s+(нарушили|нарушаете|оскорбили|использовали\s+(чит|сторон))|вы\s+будете\s+наказан|"
    r"наказание\s+(было\s+)?выдано\s+(верно|правильно|справедливо|заслуженно)|"
    r"your\s+(behaviou?r|message|actions?)\s+(violat|break|is\s+against)|you\s+(violated|broke|have\s+violated)|"
    r"(punishment|ban)\s+was\s+(correct|fair|justified)|"
    r"siz\s+qoida\w*\s+buzdingiz|сіз\s+ереже\w*\s+бұздыңыз|usted\s+(violó|infringió)|você\s+violou", _FLAGS)


def misaddressed_reply(category: str, reply: str) -> str | None:
    """Причина, если черновик адресован не тому или объявляет заявителя виновным; иначе None.
    Проверяется для жалоб (на другого игрока и апелляций), где заявитель — не установленный нарушитель."""
    if not (category.startswith("complaint.") or category == "report.complaint"):
        return None
    if category == "complaint.invalid":
        return None
    if _TELL_READER_TO_STOP.search(reply):
        return "Черновик даёт заявителю указания, как нарушителю"
    if _READER_IS_GUILTY.search(reply):
        return "Черновик объявляет заявителя виновным или решение верным"
    return None


def sensitive_reasons(player_text: str) -> list[str]:
    return [reason for reason, pattern in SENSITIVE_PATTERNS.values() if pattern.search(player_text)]


def _matches(name: str, pattern: re.Pattern, reply: str) -> bool:
    for match in pattern.finditer(reply):
        if name in _CONDITIONAL_ONLY and _CONDITIONAL.search(reply[:match.start()]):
            continue  # «если нарушение подтвердится» — не утверждение
        return True
    return False


def forbidden_in_reply(reply: str) -> list[str]:
    found = [name for name, pattern in FORBIDDEN_IN_REPLY if _matches(name, pattern, reply)]
    official = _official_contacts()
    if any(_normalize_contact(m.group(0)) not in official for m in _CONTACT.finditer(reply)):
        found.append(UNOFFICIAL_CONTACT)
    return found


_P0_CATEGORIES = {"report.account", "report.payment"}
_P0_SIGNALS = ("hacking", "payment", "threat")


def has_p0_signal(player_text: str) -> bool:
    return any(SENSITIVE_PATTERNS[name][1].search(player_text) for name in _P0_SIGNALS)


def check_priority(category: str, priority: str, player_text: str) -> tuple[str, str | None]:
    """p0 по таксономии — только взлом, платежи, угрозы вне игры (массовый баг из одного тикета не виден).
    Замер на valid показал, что LLM ставит p0 апелляциям и жалобам на админа — понижаем до p1."""
    if priority != "p0":
        return priority, None
    if category in _P0_CATEGORIES:
        return priority, None
    if any(SENSITIVE_PATTERNS[name][1].search(player_text) for name in _P0_SIGNALS):
        return priority, None
    return "p1", "Приоритет p0 понижен до p1: нет признаков взлома, платежа или угроз"


def disagreement_reason(llm_category: str, keyword_category: str | None) -> str | None:
    """Найдено в реальном тесте Ollama: модель уверенно понижала «оск» до complaint.invalid
    или report.question — категорий, где человек не нужен, — и так обходила эскалацию.
    Если ключевые слова видят нарушение, а LLM выбрала категорию без участия админа, решает человек."""
    if (keyword_category in _ADMIN_ACTION_CATEGORIES
            and llm_category not in _ADMIN_ACTION_CATEGORIES):
        return (f"LLM ({llm_category}) расходится с ключевыми словами ({keyword_category}) — "
                "проверит администратор")
    return None


def decide_needs_human(
    category: str,
    priority: str,
    llm_needs_human: bool,
    confidence: float,
    player_text: str,
    needs_human_always: list[str],
) -> tuple[bool, list[str]]:
    """Итоговое решение. LLM может только добавить необходимость человека, но не убрать её."""
    reasons = sensitive_reasons(player_text)
    if category in needs_human_always:
        reasons.append("Категория всегда требует администратора")
    if priority == "p0":
        reasons.append("Критический приоритет p0")
    if category in _ADMIN_ACTION_CATEGORIES:
        reasons.append("Решение о наказании или действие в игре подтверждает администратор")
    if confidence < LOW_CONFIDENCE:
        reasons.append("Низкая уверенность классификации")
    return bool(reasons) or llm_needs_human, reasons
