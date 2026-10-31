"""Определение языка обращения без LLM.

Тексты очень короткие («дм», «+», «osk rodni»), поэтому библиотеки определения
языка на них ошибаются. Используем простые правила:
1. Особые буквы алфавита (казахские ә, ң, ө…, узбекские ў, ҳ).
2. Для латиницы — словари частых слов (узбекский, английский, испанский,
   португальский и русский транслит).
3. Если понять нельзя — язык сервера, иначе русский (73% обращений).
"""
import re

SUPPORTED = ("ru", "kk", "uz", "en", "es", "pt")

_KK_LETTERS = set("әңөұүһқғ")
_KK_ONLY_LETTERS = set("әңөұүһ")  # «қ», «ғ» — общие с узбекской кириллицей
_KK_LATIN_LETTERS = set("äğışū")
_UZ_CYR_LETTERS = set("ўҳ")
_ES_LETTERS = set("ñ¿¡")
_PT_LETTERS = set("ãõç")

_WORDS = {
    "uz": set(
        "men meni menga manga man mani sen sizga u bu va bilan uchun nima nega nimaga "
        "qildi qilyapti qiling qilib qilmoqda bering yordam iltimos haqorat sababsiz sokdi "
        "sokvotti onadan boldi edi emas yoq bor kerak ham hech mashina pul akaunt jazo qoida "
        "qoidani oldirdi otib urdi keyin kegin ketdi menda manda rahmat salom assalomu "
        "marta marotaba ishonmaydi boshqa kirdim korip turibdi".split()
    ),
    "en": set(
        "the is are was me my you he she it this that what why how help please player "
        "killed kill insulted cheat cheater hacker stuck money car account banned unban "
        "hello hi thanks do does did run out of".split()
    ),
    "es": set(
        "el los las ayuda ayudame jugador mató mato insultó insulto mi está tengo cuenta "
        "dinero hola gracias porque pero muy".split()
    ),
    "pt": set(
        "os ajuda ajude jogador matou xingou não nao você voce minha meu conta dinheiro "
        "olá ola obrigado obrigada muito".split()
    ),
    # Русский, набранный латиницей: «osk rodni», «dm bez prichiny», «pomogite»
    "ru": set(
        "osk oskorblenie oskorbil rodni rodnyh rodnih dm db pomogite pomogi ne soglasen "
        "bez prichiny ubil menya igrok mat mamu sk tk nonrp soft softom chit chiter skam "
        "privet zastryal pozhaluysta pls".split()
    ),
}

# Кириллица без особых букв: «Ол мені тепті» (kk — есть только «і», как в украинском; «мені» есть и
# в украинском, поэтому в словарь не входит),
# «Мени сабабсиз отиб ташлади» (uz — нет ни ў, ни ҳ). Частые слова, которых нет в русском
_CYR_WORDS = {
    "kk": set(
        "менің маған сені сенің оны оның бір емес және жоқ себепсіз ойыншы әкімші ұрды "
        "атып тастады тепті бандады кісі көмек көмектесіңіз өтінемін рахмет сәлем".split()
    ),
    "uz": set(
        "мени менга менинг сени сабабсиз илтимос ёрдам нега нимага учун билан қилди қиляпти отиб урди "
        "ташлади ҳақорат жазо қоида бериб беринг рахмат ассалому".split()
    ),
}

# Узбекские апострофы: oʻ, o', gʻ, g'
_UZ_APOSTROPHE = re.compile(r"[og][ʻ'‘’`]", re.IGNORECASE)
# Служебное, что не говорит о языке: токены анонимизации, ники, номера пунктов
_NOISE = re.compile(r"<\w+>|\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b|\d+(?:\.\d+)*")


def detect_language(text: str, server_lang: str | None = None) -> tuple[str, bool]:
    """Возвращает (язык, уверены_ли_мы). Язык всегда из SUPPORTED."""
    fallback = server_lang if server_lang in SUPPORTED else "ru"
    clean = _NOISE.sub(" ", text).lower()

    cyrillic = sum(1 for ch in clean if "а" <= ch <= "я" or ch in "ёәңөұүһқғіўҳ")
    latin = sum(1 for ch in clean if "a" <= ch <= "z" or ch in "ñãõçáéíóúâêôäğışūöü")
    if cyrillic == 0 and latin == 0:
        return fallback, False

    if cyrillic >= latin:
        letters = set(clean)
        if letters & _UZ_CYR_LETTERS:
            return "uz", True
        if letters & _KK_ONLY_LETTERS:
            return "kk", True
        # «қ» и «ғ» есть и в узбекской кириллице («тиқилиб», «ёрдам беринг»): сначала словари, потом буквы
        words = set(re.findall(r"[а-яёіәңөұүһқғўҳ]+", clean))
        for lang in ("kk", "uz"):
            if words & _CYR_WORDS[lang]:
                return lang, True
        if letters & _KK_LETTERS:
            return "kk", True
        return "ru", True  # сюда же попадает украинский: отвечаем по-русски

    # Латиница
    letters = set(clean)
    if _UZ_APOSTROPHE.search(clean):
        return "uz", True
    # Казахская латиница (алфавит 2021): ä ğ ı ş ū. Проверяем до испанского — там тоже бывает ñ
    if letters & _KK_LATIN_LETTERS:
        return "kk", True
    if letters & _ES_LETTERS:
        return "es", True
    if letters & _PT_LETTERS:
        return "pt", True

    words = re.findall(r"[a-zñãõçáéíóúâêô]+", clean)
    scores = {lang: sum(1 for w in words if w in vocab) for lang, vocab in _WORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return fallback, False
    # Казахская латиница без диакритики («ol meni tepti») похожа на узбекскую: решает язык сервера
    if best == "uz" and server_lang == "kk" and not _UZ_APOSTROPHE.search(clean):
        return "kk", True
    return best, True
