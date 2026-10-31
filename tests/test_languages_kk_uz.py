"""Казахский, узбекский и «декоративный» Unicode. Без сети: LLM подменяется."""
import pytest

from app import pipeline
from app.schemas import TriageRequest
from app.services import llm
from app.services.generator import TEMPLATES
from app.services.language import detect_language
from app.services.normalize import normalize_text

OLLAMA_ENV = {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5:3b"}
GEMINI_ENV = {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-3.1-flash-lite", "GEMINI_API_KEY": "AIzaFAKE"}


@pytest.mark.parametrize("text, lang", [
    ("Ойыншы мені чатта балағаттады", "kk"),          # казахская кириллица
    ("Ол мені алдады, ақша бермеді", "kk"),
    ("Oiynşy meni çatta balağattady", "kk"),          # казахская латиница 2021 (раньше -> pt)
    ("Meniñ akkauntymdy buzdy, kömektesiñizşi", "kk"),  # ñ + ş (раньше -> es)
    ("Ойыншы маған оск родни жазды", "kk"),           # смесь с русским сленгом
    ("U meni chatda haqorat qildi", "uz"),            # узбекская латиница
    ("У мени чатда ҳақорат қилди", "uz"),             # узбекская кириллица
    ("oʻldirdi sababsiz", "uz"),
    ("Me insultó en el chat", "es"),
    ("Ele me xingou no chat, não é justo", "pt"),
])
def test_language_detection(text, lang):
    assert detect_language(normalize_text(text))[0] == lang


@pytest.mark.parametrize("raw, clean", [
    ("Ｏｓｋ ｒｏｄｎｉ", "Osk rodni"),        # полноширинные буквы
    ("𝐨𝐬𝐤 𝐫𝐨𝐝𝐧𝐢", "osk rodni"),                # математические жирные
    ("о​с​к родни", "оск родни"),    # zero-width внутри слова
    ("2​.1 оп", "2.1 оп"),
])
def test_unicode_normalization(raw, clean):
    assert normalize_text(raw) == clean


def test_decorated_text_is_classified():
    out = pipeline.triage(TriageRequest(kind="complaint", messages=[{"role": "player", "text": "𝐨𝐬𝐤 𝐫𝐨𝐝𝐧𝐢 2​.1",
                                                                      "attachments": 1}]))
    assert out.category == "complaint.insult" and out.extracted.rule_refs == ["2.1"]


# ---------- Черновики по языкам ----------

def _fake(reply):
    return lambda **kw: llm.LLMCall(result=llm.LLMResult(
        category="complaint.insult", priority="p2", cited=["2.1"], reply=reply, needs_human=True,
        confidence=0.9, reason="тест"), cost_usd=0.0, latency_s=0.1)


def _kk_complaint():
    return TriageRequest(kind="complaint", messages=[{"role": "player", "attachments": 1,
                         "text": "Бұл ойыншы кеше кешке чатта менің отбасымды ұзақ балағаттады"}])


def test_ollama_kk_draft_replaced_by_template(monkeypatch):
    """Реальный тест qwen2.5:3b: казахские черновики бессвязные -> для Ollama по умолчанию шаблон."""
    import httpx
    from app.services import llm_providers
    monkeypatch.setattr(llm_providers.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")))
    llm.init(OLLAMA_ENV)
    monkeypatch.setattr(llm, "classify", _fake("Жақссы жазууға болады, бірақ тексеріңіз."))
    out = pipeline.triage(_kk_complaint())
    assert out.lang == "kk"
    assert out.draft_reply.text == TEMPLATES["complaint_received"]["kk"]
    assert any("Ollama (эксперимент" in r for r in out.reasons)  # Ollama только подсказывает


def test_gemini_kk_draft_allowed(monkeypatch):
    llm.init(GEMINI_ENV, synthetic_data=True)
    monkeypatch.setattr(llm, "classify", _fake("Сәлеметсіз бе! Шағымыңыз әкімшіге жіберілді."))
    assert pipeline.triage(_kk_complaint()).draft_reply.text.startswith("Сәлеметсіз бе! Шағымыңыз әкімшіге")


def test_reply_in_wrong_language_replaced(monkeypatch):
    llm.init(GEMINI_ENV, synthetic_data=True)
    monkeypatch.setattr(llm, "classify", _fake("Здравствуйте! Ваша жалоба передана администратору."))
    out = pipeline.triage(_kk_complaint())  # игрок пишет по-казахски, модель ответила по-русски
    assert out.draft_reply.text == TEMPLATES["complaint_received"]["kk"]
    assert any("вместо kk" in r for r in out.reasons)


def test_draft_langs_config(monkeypatch):
    from app.services.llm_config import LLMConfigError, load_settings
    assert load_settings({**OLLAMA_ENV, "LLM_DRAFT_LANGS": "ru,uz"}).draft_langs == {"ru", "uz"}
    with pytest.raises(LLMConfigError):
        load_settings({**OLLAMA_ENV, "LLM_DRAFT_LANGS": "ru,de"})


# Кириллица без особых букв: раньше уходила в русский (найдено при ручной проверке 2026-10-31)
@pytest.mark.parametrize("text, lang", [
    ("Ол мені тепті", "kk"),
    ("Мені себепсіз атып тастады", "kk"),
    ("Админ мені себепсіз бандады", "kk"),
    ("Мени сабабсиз отиб ташлади", "uz"),
    ("Привіт, допоможіть мені", "ru"),   # украинский: «мені» не считается казахским
    ("мне дали бан без причины", "ru"),
])
def test_cyrillic_without_special_letters(text, lang):
    assert detect_language(text)[0] == lang


def test_latin_kk_without_diacritics_uses_server_lang():
    assert detect_language("ol meni tepti", "kk")[0] == "kk"
    assert detect_language("ol meni tepti")[0] == "uz"
    assert detect_language("o'yinchi meni haqorat qildi", "kk")[0] == "uz"  # апостроф — точно узбекский


# Категории kk/uz без LLM (слова подобраны по пропускам в tickets_train, проверены на valid)
@pytest.mark.parametrize("text, category", [
    ("menga be sabab mute berishdi", "complaint.admin"),        # «без причины» + наказание → апелляция
    ("admin meni sababsiz qamadi", "complaint.admin"),
    ("Маған себепсіз бан берді, келіспеймін", "complaint.admin"),
    ("Менга ноҳақ бан беришди", "complaint.admin"),
    ("u meni sababsiz otib tashladi", "complaint.dm"),              # «без причины» + убийство → DM
    ("Ол мені себепсіз атып тастады", "complaint.dm"),
    ("meni soʻkdi", "complaint.insult"),                             # узбекский апостроф ʻ
    ("u meni haromi dedi", "complaint.insult"),
    ("Бу ўйинчи онамни ҳақорат қилди", "complaint.insult"),
])
def test_kk_uz_keyword_categories(text, category):
    from app.services.classifier import classify_by_keywords
    result = classify_by_keywords("complaint", text, [])
    assert result is not None and result.category == category


@pytest.mark.parametrize("text", [
    "U meni bank oldida sababsiz otib tashladi",
    "У мени банк олдида сабабсиз отиб ташлади",
    "Ол мені банктің жанында себепсіз атып тастады",
])
def test_bank_is_not_ban(text):  # «банк» начинается с «бан» — это DM, а не апелляция
    from app.services.classifier import classify_by_keywords
    assert classify_by_keywords("complaint", text, []).category == "complaint.dm"


def test_ollama_drafts_are_templates_by_default():
    from app.services.llm_config import load_settings
    assert load_settings(OLLAMA_ENV).draft_langs == frozenset()  # слепая оценка: шаблоны лучше черновиков qwen2.5:3b
