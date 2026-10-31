from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REQUIRED_FIELDS = {
    "lang", "category", "priority", "is_spam_or_duplicate", "extracted",
    "draft_reply", "needs_human", "confidence", "reasons",
}


def test_health():
    assert client.get("/health").json() == {"status": "ok", "llm_provider": "disabled"}


def test_contract_example():
    body = {
        "kind": "complaint",
        "messages": [{"role": "player", "text": "Игрок оскорблял меня, 2.1 ОП", "attachments": 2}],
    }
    response = client.post("/triage", json=body)
    assert response.status_code == 200
    data = response.json()
    assert REQUIRED_FIELDS <= data.keys()
    assert data["lang"] == "ru"
    assert data["extracted"]["rule_refs"] == ["2.1"]
    assert data["extracted"]["has_evidence"] is True
    assert data["draft_reply"]["lang"] == data["lang"]


def test_empty_report_is_spam():
    body = {"kind": "report", "messages": [{"role": "player", "text": "+"}]}
    data = client.post("/triage", json=body).json()
    assert data["is_spam_or_duplicate"] is True
    assert data["category"] == "report.offtopic"


def test_rules_endpoint():
    data = client.get("/rules", params={"ids": ["2.1", "GZ 1.2", "99.9"]}).json()
    assert [r["id"] for r in data] == ["2.1", "GZ 1.2"]
    assert data[0]["text"]


def test_bad_request_rejected():
    assert client.post("/triage", json={"kind": "complaint", "messages": []}).status_code == 422
    assert client.post("/triage", json={"kind": "other", "messages": [{"text": "x"}]}).status_code == 422


def test_oversized_fields_rejected():
    one = {"role": "player", "text": "оск"}
    too_many = {"kind": "complaint", "messages": [one] * 51}
    assert client.post("/triage", json=too_many).status_code == 422
    long_text = {"kind": "complaint", "messages": [{"role": "player", "text": "a" * 5001}]}
    assert client.post("/triage", json=long_text).status_code == 422
    long_nick = {"kind": "complaint", "messages": [one], "reported_nick": "x" * 65}
    assert client.post("/triage", json=long_nick).status_code == 422
