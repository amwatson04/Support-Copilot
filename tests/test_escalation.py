import pytest

from app import pipeline
from app.schemas import TriageRequest
from app.services import llm
from app.services.escalation import forbidden_in_reply, sensitive_reasons


@pytest.mark.parametrize("text", [
    "не согласен с наказанием", "опру", "прошу пересмотреть блокировку", "снимите бан пожалуйста",
    "разбаньте", "требую доказательства", "ne sogl",
])
def test_appeals_detected(text):
    assert any("Апелляция" in r for r in sensitive_reasons(text))


@pytest.mark.parametrize("text", ["меня взломали", "my account was hacked", "угрожает взломом"])
def test_hacking_detected(text):
    assert any("взлом" in r for r in sensitive_reasons(text))


@pytest.mark.parametrize("text", ["оплатил гк, не пришло", "деньги списали", "донат не пришел", "payment failed"])
def test_payment_detected(text):
    assert any("оплаты" in r for r in sensitive_reasons(text))


@pytest.mark.parametrize("text", ["оск родни", "дм у банка", "помогите застрял", "как устроиться в полицию"])
def test_ordinary_texts_not_sensitive(text):
    assert sensitive_reasons(text) == []


@pytest.mark.parametrize("reply", [
    "Мы вас разбаним в ближайшее время", "Вернём деньги в течение дня", "Игрок будет наказан",
    "На скриншоте видно нарушение", "Your ban will be lifted, we will refund you", "The player will be banned",
    # из реального теста Ollama qwen2.5:3b, 2026-10-31
    "Ваша жалоба на оскорбления подтверждается.",
    "Если подтверждено нарушение, будет принято соответствующее наказание.",
    "Пожалуйста, свяжитесь с поддержкой по адресу support@grandmobile.com.",
    "Подробнее: https://example.com/rules", "Пишите в t.me/some_support", "Позвоните +7 701 123 45 67",
])
def test_forbidden_promises_found(reply):
    assert forbidden_in_reply(reply)


@pytest.mark.parametrize("reply", [
    "Здравствуйте! Жалоба передана администрации для проверки.",
    "Пожалуйста, приложите видео нарушения согласно правилам подачи жалоб.",
    "Жалоба передана администратору, он проверит доказательства и примет решение.",
    "Ожидайте ответа в течение 24 часов, пункт правил 2.1.",
    # официальная ссылка из правил жалоб (ЖБ 1.1.2) — разрешена
    "Для технических вопросов обратитесь в техническую поддержку: https://support.grnd.gg/",
])
def test_normal_replies_pass(reply):
    assert forbidden_in_reply(reply) == []


def _fake(**fields):
    defaults = dict(category="report.question", priority="p3", cited=[], reply="Ответ", needs_human=False,
                    confidence=0.9, reason="Вопрос")
    defaults.update(fields)
    return lambda **kwargs: llm.LLMCall(result=llm.LLMResult(**defaults), cost_usd=0.0, latency_s=0.1)


def _run(text, kind="report"):
    return pipeline.triage(TriageRequest(kind=kind, messages=[{"role": "player", "text": text, "attachments": 1}]))


def test_llm_cannot_skip_human_for_appeal(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.other", needs_human=False))
    out = _run("не согласен с наказанием, снимите варн", kind="complaint")
    assert out.needs_human is True


def test_bad_reply_replaced_with_template(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.insult", cited=["2.1"],
                                               reply="Нарушитель будет наказан, на скриншоте видно оскорбление"))
    out = _run("Этот игрок долго писал гадости про мою семью в общий чат", kind="complaint")
    assert "наказан" not in out.draft_reply.text
    assert out.draft_reply.cited == []
    assert out.needs_human is True
    assert any("заменён шаблоном" in r for r in out.reasons)


def test_simple_question_can_be_auto(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake())
    out = _run("как устроиться в полицию")
    assert out.needs_human is False


def test_p0_lowered_for_appeal(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.admin", priority="p0", needs_human=True))
    out = _run("Не согласен с наказанием, администратор выдал его ошибочно вчера", kind="complaint")
    assert out.priority == "p2"  # p0 без признаков взлома снят; дальше — приоритет категории complaint.admin
    assert out.needs_human is True


def test_p0_kept_for_hacking(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(category="report.account", priority="p0", needs_human=True))
    assert _run("меня взломали, пропала машина").priority == "p0"


def test_low_confidence_goes_to_human(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(confidence=0.3))
    assert _run("подскажите пожалуйста как устроиться работать в полицию города").needs_human is True


def test_llm_cannot_downgrade_to_avoid_human(monkeypatch):
    """Реальный случай Ollama: «оск…» в длинном тексте → модель уверенно ставит complaint.invalid."""
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.invalid", priority="p3", needs_human=False,
                                               confidence=1.0))
    out = _run("он меня оскорблял весь вечер в чате и не останавливался вообще", kind="complaint")
    assert out.needs_human is True
    assert any("расходится" in r for r in out.reasons)


def test_agreeing_llm_question_stays_auto(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake())  # report.question, без человека
    out = _run("подскажите пожалуйста как устроиться работать в полицию города")
    assert out.needs_human is False


# ---------- Слепая оценка черновиков 2026-10-31: новые типы ошибок ----------

@pytest.mark.parametrize("reply", [
    "Сожалеем об этом инциденте. Ваш аккаунт будет восстановлен после рассмотрения.",
    "I have noted your complaint. This behavior is clearly a cheat.",
    "Ваш платеж был успешно обработан, но верификация может занять время.",
])
def test_new_blind_eval_errors_caught(reply):
    assert forbidden_in_reply(reply)


@pytest.mark.parametrize("reply", [
    "An administrator will review the evidence and take appropriate action if a violation is confirmed.",
    "Администратор примет меры, если нарушение будет подтверждено.",
    "Ваша заявка по поводу платежа принята в работу, ожидайте проверки.",
])
def test_conditional_and_neutral_replies_pass(reply):
    assert forbidden_in_reply(reply) == []


def test_disagreement_replaces_draft(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.invalid", priority="p3", needs_human=False,
                                               reply="Ваше сообщение не содержит достаточной информации."))
    out = _run("Игрок убил меня у банка без причины, просто подошёл и выстрелил", kind="complaint")
    assert "недостаточной" not in out.draft_reply.text
    assert out.needs_human is True and any("спорная" in r for r in out.reasons)


def test_llm_priority_follows_category_rules(monkeypatch):
    # qwen2.5:3b ставила p1 оскорблениям; приоритет берётся из правил категории
    monkeypatch.setattr(llm, "classify", _fake(category="complaint.insult", priority="p1"))
    out = _run("Этот игрок весь вечер писал гадости про мою семью в общий чат", kind="complaint")
    assert out.priority == "p2" and any("по правилам категории" in r for r in out.reasons)


def test_draft_saying_punishment_was_wrong_is_replaced(monkeypatch):
    monkeypatch.setattr(llm, "classify", _fake(
        category="complaint.admin", reply="Сожалеем, что вы получили наказание по необоснованным причинам."))
    out = _run("мне неверно выдан варн, прошу снять, я ничего не делал", kind="complaint")
    assert "необоснованн" not in out.draft_reply.text and out.needs_human


def test_no_p0_from_category_without_hacking_signs(monkeypatch):
    # qwen2.5:3b назвала смену ника взломом (report.account): p0 только при признаках взлома/платежа/угроз
    monkeypatch.setattr(llm, "classify", _fake(category="report.account", priority="p2", needs_human=True))
    assert _run("смена нон рп ника на другой, проверьте пожалуйста этого игрока").priority == "p2"


def test_latin_tekstur_is_help():
    from app.services.classifier import classify_by_keywords
    assert classify_by_keywords("report", "Men teksturağa tūryp qaldym, şyğa almaimyn", []).category == "report.help"


def test_draft_confirming_cheat_use_is_forbidden():
    from app.services.escalation import forbidden_in_reply
    assert forbidden_in_reply("Ваше сообщение подтверждает использование читов и стороннего ПО.")
    assert not forbidden_in_reply("Администратор проверит, подтверждается ли нарушение.")  # нейтрально — можно


@pytest.mark.parametrize("reply", [
    "Ваше сообщение не содержит достаточной информации. Жалоба будет отклонена без рассмотрения.",
    "Ваша жалоба будет рассмотрена. Максимальная санкция - блокировка персонажа на 10 дней.",
])
def test_draft_deciding_outcome_is_forbidden(reply):
    from app.services.escalation import forbidden_in_reply
    assert forbidden_in_reply(reply)


@pytest.mark.parametrize("text", [
    "персонаж заблокирован администратором без проверки, я не использовал читы",
    "я ничего не делал, зашто мут дали",
])
def test_more_appeal_phrases(text):
    from app.services.escalation import SENSITIVE_PATTERNS
    assert SENSITIVE_PATTERNS["appeal"][1].search(text)
