"""predict.py: формат вывода, продолжение с места остановки, ожидание при 429. Без сети."""
import json
import sys

import pytest

from app import pipeline
from app.services import llm
from app.services.deduplication import DuplicateDetector
from scripts import predict


def _record(i, text="Игрок вчера вёл себя странно рядом с банком, разберитесь пожалуйста", kind="complaint"):
    return {"ticket_id": f"t{i}", "kind": kind, "created_at": f"2026-10-01T12:{i:02d}:00",
            "reporter_hash": f"r{i}", "reported_nick": "Arman_Petrov", "server_lang": "ru", "images_count": 1,
            "messages": [{"role": "player", "text": text}, {"role": "admin", "text": "ответ админа"}]}


@pytest.fixture
def files(tmp_path, monkeypatch):
    source = tmp_path / "in.jsonl"
    source.write_text("\n".join(json.dumps(_record(i), ensure_ascii=False) for i in range(5)), encoding="utf-8")
    target = tmp_path / "out.jsonl"
    monkeypatch.setattr(pipeline, "duplicates", DuplicateDetector())
    monkeypatch.setattr(predict, "QUOTA_WAITS_S", (0,))

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["predict", "--input", str(source), "--output", str(target),
                                          "--provider", "disabled", "--rpm", "0", *extra])
        predict.main()
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    return run


def test_output_has_contract_fields(files):
    rows = files()
    assert len(rows) == 5
    assert {"ticket_id", "lang", "category", "priority", "is_spam_or_duplicate", "extracted",
            "draft_reply", "needs_human", "confidence", "reasons"} <= rows[0].keys()


def test_resume_skips_done_tickets(files):
    assert len(files("--limit", "2")) == 2
    rows = files()
    assert [r["ticket_id"] for r in rows] == ["t0", "t1", "t2", "t3", "t4"]  # без повторов


def test_quota_retries_same_ticket(files, monkeypatch):
    answers = [llm.LLMFailure("quota")]  # первый вызов — 429, дальше успех

    def fake(**kwargs):
        if answers:
            return answers.pop()
        return llm.LLMCall(result=llm.LLMResult(category="complaint.insult", priority="p2", cited=["2.1"],
                                                reply="Черновик", needs_human=True, confidence=0.9,
                                                reason="Оскорбление"), cost_usd=0.0, latency_s=0.1)
    monkeypatch.setattr(llm, "classify", fake)
    rows = files()
    assert len(rows) == 5
    assert rows[0]["category"] == "complaint.insult"
    assert rows[0]["is_spam_or_duplicate"] is False  # повтор после 429 — не дубликат самого себя


def test_persistent_quota_stops_and_keeps_progress(files, monkeypatch):
    monkeypatch.setattr(llm, "classify", lambda **kwargs: llm.LLMFailure("quota"))
    with pytest.raises(SystemExit) as stop:
        files()
    assert stop.value.code == 2


def test_fresh_recomputes_everything(files, capsys):
    files("--limit", "2")
    capsys.readouterr()
    rows = files("--fresh")
    assert len(rows) == 5  # файл очищен и пересчитан, без дублей
    assert "ВНИМАНИЕ" not in capsys.readouterr().out


def test_resume_warns_about_stale_results(files, capsys):
    files("--limit", "2")
    capsys.readouterr()
    files()
    assert "НЕ пересчитываются" in capsys.readouterr().out
