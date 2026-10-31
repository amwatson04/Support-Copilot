from datetime import datetime, timedelta

from app.services.deduplication import DuplicateDetector, is_spam
from app.services.language import detect_language


def test_language_by_letters():
    assert detect_language("Игрок оскорблял меня") == ("ru", True)
    assert detect_language("Ойыншы мені чатта балағаттады") == ("kk", True)
    assert detect_language("Ўзбекча матн, ҳақорат") == ("uz", True)
    assert detect_language("¿Por qué me banearon?")[0] == "es"
    assert detect_language("Não consigo entrar na conta")[0] == "pt"


def test_language_by_words():
    assert detect_language("U meni chatda haqorat qildi")[0] == "uz"
    assert detect_language("help me please, I am stuck")[0] == "en"
    assert detect_language("osk rodni")[0] == "ru"  # русский транслитом
    assert detect_language("oʻldirdi")[0] == "uz"


def test_language_fallback():
    assert detect_language("+") == ("ru", False)
    assert detect_language("+", server_lang="uz") == ("uz", False)
    assert detect_language("<url> 2.1") == ("ru", False)


def test_spam():
    assert is_spam("", 0)
    assert is_spam("+", 0)
    assert is_spam("53", 0)
    assert is_spam("ааааа", 0)
    assert not is_spam("+", 2)           # есть скриншоты
    assert not is_spam("<url>", 0)       # есть ссылка на видео
    assert not is_spam("2.1", 0)         # указан пункт правил
    assert not is_spam("помогите", 0)    # просьба о помощи — не спам


def test_duplicates():
    detector = DuplicateDetector()
    t = datetime(2026, 10, 1, 12, 0)
    assert not detector.check_and_remember("u1", "Arman_Petrov", "оск родни", t)
    assert detector.check_and_remember("u1", "arman_petrov", "Оск  родни!", t + timedelta(hours=1))
    assert not detector.check_and_remember("u1", "Arman_Petrov", "дм у банка", t + timedelta(hours=2))
    assert not detector.check_and_remember("u2", "Arman_Petrov", "оск родни", t)       # другой заявитель
    assert not detector.check_and_remember("u1", "Arman_Petrov", "оск родни", t + timedelta(days=3))
    assert not detector.check_and_remember(None, "Arman_Petrov", "оск родни", t)       # нет данных
