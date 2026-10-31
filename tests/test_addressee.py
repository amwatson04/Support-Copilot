"""Адресат черновика: ответ читает заявитель, а не нарушитель.

Реальные ошибки, на которых построены тесты (2026-10-31):
- Gemini, жалоба на оскорбления: «…Пожалуйста, воздержитесь от использования ненормативной лексики»;
- Ollama, жалоба на читы (en): «Please refrain from using any unauthorized software»;
- Ollama, апелляция: «Ваше поведение нарушает правила игры».
"""
import pytest

from app import pipeline
from app.schemas import TriageRequest
from app.services import llm
from app.services.escalation import misaddressed_reply
from app.services.generator import TEMPLATES, template_for


# ---------- Черновики, адресованные не тому ----------

@pytest.mark.parametrize("category, reply", [
    # жалоба от первого лица: «он меня оскорбил»
    ("complaint.insult", "Ваша жалоба принята. Пожалуйста, воздержитесь от использования ненормативной лексики."),
    ("complaint.insult", "Пожалуйста, извинитесь перед игроком и будьте вежливы в чате."),
    # жалоба в третьем лице: «игрок X использует читы»
    ("complaint.cheat", "Please refrain from using any unauthorized software."),
    ("complaint.cheat", "Не используйте стороннее ПО, это нарушение правил."),
    ("complaint.dm", "Прекратите убивать игроков без причины."),
    # прямое обвинение заявителя
    ("complaint.insult", "Ваше сообщение содержит оскорбления."),
    ("report.complaint", "Вы нарушили правила сервера."),
    # апелляция: заявитель — наказанный игрок, вина и решение не установлены черновиком
    ("complaint.admin", "Ваше поведение нарушает правила игры."),
    ("complaint.admin", "Наказание выдано верно, жалоба закрыта."),
    ("complaint.admin", "The ban was correct, you violated the rules."),
    # другие языки
    ("complaint.insult", "Iltimos, haqorat qilmang."),
    ("complaint.insult", "Балағаттамаңыз, бұл ережеге қайшы."),
    ("complaint.cheat", "Por favor, deje de usar trampas."),
    ("complaint.cheat", "Por favor, pare de usar trapaças."),
])
def test_misaddressed_drafts_are_caught(category, reply):
    assert misaddressed_reply(category, reply)


@pytest.mark.parametrize("category, reply", [
    ("complaint.insult", "Спасибо за жалобу. Администратор проверит доказательства и примет решение."),
    ("complaint.cheat", "Thank you for the report. An administrator will review the video."),
    ("complaint.admin", "Ваша апелляция будет рассмотрена администратором."),
    ("complaint.dm", "Жалоба на убийство без причины передана администратору, пункт правил 5.3."),
    # неоднозначное обращение, но нейтральный ответ — допустим
    ("complaint.other", "Здравствуйте! Мы передали обращение администратору для проверки."),
    # не жалобы — проверка адресата не применяется
    ("report.help", "Не используйте этот баг повторно, администратор поможет выбраться."),
    ("complaint.invalid", "Пожалуйста, опишите нарушение подробнее и приложите доказательства."),
])
def test_correctly_addressed_drafts_pass(category, reply):
    assert misaddressed_reply(category, reply) is None


# ---------- Шаблоны по категории ----------

def test_template_by_category():
    assert template_for("complaint.insult") == "complaint_received"
    assert template_for("report.complaint") == "complaint_received"
    assert template_for("complaint.admin") == "appeal_received"
    assert template_for("report.help") == "received"


def test_template_by_text_and_evidence():
    # Апелляция, пришедшая как report или с категорией нарушения, — не «спасибо за жалобу»
    assert template_for("report.complaint", "Не согласен с наказанием, меня забанили") == "appeal_received"
    assert template_for("complaint.insult", "я не оскорблял, снимите мут") == "appeal_received"
    # Без скриншотов и ссылок не пишем «приложенные доказательства»
    assert template_for("complaint.cheat", "у него софт", has_evidence=False) == "complaint_no_evidence"
    assert template_for("complaint.cheat", "у него софт", has_evidence=True) == "complaint_received"


def test_no_evidence_template_does_not_claim_evidence():
    for lang in ("ru", "kk", "uz", "en", "es", "pt"):
        assert TEMPLATES["complaint_no_evidence"][lang] != TEMPLATES["complaint_received"][lang]
    assert "приложенные" not in TEMPLATES["complaint_no_evidence"]["ru"]


def test_pipeline_without_evidence_and_appeal_as_report():
    out = pipeline.triage(TriageRequest(kind="complaint", messages=[{"text": "у него софт, аим летает"}]))
    assert out.draft_reply.text == TEMPLATES["complaint_no_evidence"]["ru"]
    out = pipeline.triage(TriageRequest(kind="report", messages=[
        {"text": "Не согласен с наказанием, меня забанили ни за что"}]))
    assert out.draft_reply.text == TEMPLATES["appeal_received"]["ru"] and out.needs_human


@pytest.mark.parametrize("name", ["complaint_received", "complaint_no_evidence", "appeal_received"])
def test_templates_exist_in_all_languages_and_are_clean(name):
    for lang in ("ru", "kk", "uz", "en", "es", "pt"):
        text = TEMPLATES[name][lang]
        assert text and misaddressed_reply("complaint.insult", text) is None


# ---------- Полный путь через pipeline ----------

def _fake(**fields):
    defaults = dict(category="complaint.insult", priority="p2", cited=["2.1"], needs_human=True,
                    confidence=0.9, reason="тест",
                    reply="Ваша жалоба принята. Пожалуйста, воздержитесь от использования ненормативной лексики.")
    defaults.update(fields)
    return lambda **kwargs: llm.LLMCall(result=llm.LLMResult(**defaults), cost_usd=0.0, latency_s=0.1)


def _run(text, kind="complaint"):
    return pipeline.triage(TriageRequest(kind=kind, messages=[{"role": "player", "text": text, "attachments": 1}]))


def test_pipeline_replaces_misaddressed_complaint_draft(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake())
    out = _run("Этот игрок весь вечер писал гадости про мою семью в общий чат")
    assert out.draft_reply.text == TEMPLATES["complaint_received"]["ru"]
    assert out.needs_human is True and out.draft_reply.cited == []
    assert any("указания" in r for r in out.reasons)


def test_pipeline_replaces_guilty_appeal_draft_in_player_language(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.admin",
                                               reply="Your behaviour violates the rules of the game."))
    out = _run("I think my ban was a mistake, please review the situation again")
    assert out.lang == "en"
    assert out.draft_reply.text == TEMPLATES["appeal_received"]["en"]


def test_keyword_path_uses_addressed_template():
    out = _run("оск родни")  # без LLM (conftest: disabled), короткая жалоба
    assert out.draft_reply.text == TEMPLATES["complaint_received"]["ru"]
    appeal = _run("опру")
    assert appeal.draft_reply.text == TEMPLATES["appeal_received"]["ru"]


# Слепая оценка этапа 2 (Ollama, 2026-10-31)
@pytest.mark.parametrize("reply", [
    "Ваше поведение в чате нарушает правила игры. Пожалуйста, соблюдайте этичное поведение.",
    "Ваш никнейм не соответствует правилам игры. Пожалуйста, используйте разрешенные никнеймы.",
])
def test_stage2_misaddressed_drafts_caught(reply):
    assert misaddressed_reply("complaint.insult", reply)


def test_victim_complaint_called_appeal_by_llm_gets_complaint_template():
    # пострадавший просит наказать обидчика; LLM сказала complaint.admin
    assert template_for("complaint.admin", "u onamni so'kdi, uni ban qiling",
                        keyword_category="complaint.insult") == "complaint_received"
    assert template_for("complaint.admin", "не согласен с наказанием", keyword_category="complaint.insult") == "appeal_received"
    assert template_for("complaint.admin", "Menga nohaq ban berishdi") == "appeal_received"
