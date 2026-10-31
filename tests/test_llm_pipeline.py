"""Тесты пайплайна с LLM. Настоящую LLM не вызываем: подменяем ответ (monkeypatch).
Так тесты бесплатные, быстрые и работают без интернета и ключа.

Короткие однозначные тексты («оск») классифицируются без LLM, поэтому
для проверки пути через LLM используем длинные тексты."""
from app import pipeline
from app.schemas import TriageRequest
from app.services import llm

LONG_INSULT = "Этот игрок долго оскорблял мою семью в общем чате, 2.1 ОП, прошу разобраться"
NEUTRAL = "Игрок вчера вёл себя странно рядом с банком, разберитесь пожалуйста"


def make_request(text: str, kind: str = "complaint", attachments: int = 1) -> TriageRequest:
    return TriageRequest(kind=kind, messages=[{"role": "player", "text": text, "attachments": attachments}])


def fake_llm(**fields):
    defaults = dict(category="complaint.insult", priority="p2", cited=["2.1"], reply="Черновик",
                    needs_human=True, confidence=0.8, reason="Оскорбление")
    defaults.update(fields)

    def _classify(**kwargs):
        return llm.LLMCall(result=llm.LLMResult(**defaults), cost_usd=0.0, latency_s=1.0)
    return _classify


def boom(**kwargs):
    raise AssertionError("LLM не должна вызываться")


def test_llm_answer_used(monkeypatch):
    monkeypatch.setattr(llm, "classify", fake_llm())
    out = pipeline.triage(make_request(LONG_INSULT))
    assert out.category == "complaint.insult"
    assert out.draft_reply.cited == ["2.1"]


def test_invented_citation_removed(monkeypatch):
    monkeypatch.setattr(llm, "classify", fake_llm(cited=["2.1", "99.9", "5.5"]))  # 5.5 есть, но не кандидат
    out = pipeline.triage(make_request(LONG_INSULT))
    assert out.draft_reply.cited == ["2.1"]
    assert any("отброшена" in r for r in out.reasons)


def test_payment_always_human(monkeypatch):
    monkeypatch.setattr(llm, "classify", fake_llm(category="report.payment", priority="p1", needs_human=False, cited=[]))
    out = pipeline.triage(make_request("оплатил гк через карту вчера вечером, но так и не пришло", kind="report",
                                       attachments=0))
    assert out.needs_human is True


def test_wrong_kind_category_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "classify", fake_llm(category="report.help"))
    out = pipeline.triage(make_request(NEUTRAL))
    assert out.needs_human is True and out.confidence == 0.0


def test_llm_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "classify", lambda **kwargs: llm.LLMFailure("quota"))
    out = pipeline.triage(make_request(NEUTRAL))
    assert out.needs_human is True
    assert out.category == "complaint.other"
    assert any("429" in r for r in out.reasons)


def test_llm_unavailable_uses_keywords(monkeypatch):
    monkeypatch.setattr(llm, "classify", lambda **kwargs: llm.LLMFailure("timeout"))
    out = pipeline.triage(make_request(LONG_INSULT))
    assert out.category == "complaint.insult"
    assert out.needs_human is True
    assert any("ключевым словам" in r for r in out.reasons)


def test_disabled_provider_gives_safe_answer():
    out = pipeline.triage(make_request(LONG_INSULT))  # conftest: LLM_PROVIDER=disabled
    assert out.needs_human is True
    assert out.extracted.rule_refs == ["2.1"]           # логика без LLM продолжает работать
    assert out.category == "complaint.insult"
    assert any("отключена" in r for r in out.reasons)


def test_short_clear_text_skips_llm(monkeypatch):
    monkeypatch.setattr(llm, "classify", boom)
    assert pipeline.triage(make_request("оск родни")).category == "complaint.insult"
    assert pipeline.triage(make_request("опру")).category == "complaint.admin"
    out = pipeline.triage(make_request("помогите", kind="report", attachments=0))
    assert out.category == "report.offtopic" and out.needs_human is False
    assert pipeline.triage(make_request("я застрял", kind="report", attachments=0)).category == "report.help"
    teleport = pipeline.triage(make_request("телепортируйте меня", kind="report", attachments=0))
    assert teleport.category == "report.offtopic" and teleport.needs_human is False


def test_spam_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(llm, "classify", boom)
    out = pipeline.triage(make_request("+", kind="report", attachments=0))
    assert out.is_spam_or_duplicate is True
