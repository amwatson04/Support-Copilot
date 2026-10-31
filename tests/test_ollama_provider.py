"""OllamaProvider и переключение провайдеров — НА MOCK-ОТВЕТАХ HTTP.

Настоящая Ollama здесь не запускается: подменяем httpx.get/httpx.post.
Эти тесты проверяют интерфейс, формат запроса и обработку ошибок,
но НЕ качество и скорость реальной модели.
"""
import json

import httpx
import pytest

from app import pipeline
from app.schemas import TriageRequest
from app.services import llm, llm_providers
from app.services.llm_config import LLMConfigError, load_settings
from app.services.rules import get_rule

OLLAMA_ENV = {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5:3b"}
URL = "http://localhost:11434"
VALID = {"category": "complaint.insult", "priority": "p2", "cited": ["2.1"], "reply": "Черновик",
         "needs_human": True, "confidence": 0.8, "reason": "Оскорбление"}


def ollama_reply(content, status=200, **extra) -> httpx.Response:
    body = {"message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop",
            "prompt_eval_count": 1800, "eval_count": 90, **extra}
    return httpx.Response(status, json=body, request=httpx.Request("POST", f"{URL}/api/chat"))


@pytest.fixture
def ollama(monkeypatch):
    """Включает провайдер ollama; все HTTP-запросы к нему подменены."""
    tags = {"models": [{"name": "qwen2.5:3b", "model": "qwen2.5:3b"}]}
    monkeypatch.setattr(llm_providers.httpx, "get",
                        lambda url, **kw: httpx.Response(200, json=tags, request=httpx.Request("GET", url)))
    llm.init(OLLAMA_ENV)
    sent, outcomes = [], []

    def fake_post(url, json=None, timeout=None):
        sent.append({"url": url, "json": json, "timeout": timeout})
        outcome = outcomes.pop(0) if outcomes else ollama_reply(_dumps(VALID))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm_providers.httpx, "post", fake_post)
    return sent, outcomes


def _dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False)


def _classify():
    return llm.classify(kind="complaint", text="оск родни", lang="ru", attachments=1,
                        player_refs=[], rules=[get_rule("2.1")])


# ---------- Конфигурация ----------

def test_settings_defaults_and_aliases():
    settings = load_settings({**OLLAMA_ENV, "OLLAMA_BASE_URL": "http://127.0.0.1:11434/"})
    assert settings.provider == "ollama" and settings.model == "qwen2.5:3b"
    assert settings.ollama_url == "http://127.0.0.1:11434"
    assert settings.ollama_num_ctx == 4096  # больше дефолтных 2048 Ollama
    assert load_settings({**OLLAMA_ENV, "OLLAMA_URL": "http://localhost:1"}).ollama_url == "http://localhost:1"  # старое имя


def test_remote_ollama_rejected():
    # «Локальная» модель на чужом сервере — это отправка реальных тикетов наружу
    with pytest.raises(LLMConfigError, match="localhost"):
        load_settings({**OLLAMA_ENV, "OLLAMA_BASE_URL": "http://192.0.2.10:11434"})


def test_ollama_does_not_change_answer(monkeypatch):
    """Решение 2026-10-31: Ollama — эксперимент; категория, приоритет, needs_human и черновик — как без LLM."""
    text = "Этот игрок долго оскорблял мою семью в общем чате, 2.1 ОП"
    llm.init({"LLM_PROVIDER": "disabled"})
    baseline = pipeline.triage(_request(text))
    monkeypatch.setattr(llm_providers.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")))
    llm.init(OLLAMA_ENV)
    monkeypatch.setattr(llm, "classify", lambda **kw: llm.LLMCall(result=llm.LLMResult(
        category="complaint.invalid", priority="p3", cited=[], reply="Ваше поведение нарушает правила",
        needs_human=False, confidence=0.99, reason="тест"), cost_usd=0.0, latency_s=0.1))
    pipeline.duplicates = type(pipeline.duplicates)()
    out = pipeline.triage(_request(text))
    for field in ("category", "priority", "needs_human", "draft_reply"):
        assert getattr(out, field) == getattr(baseline, field), field
    assert any("Ollama (эксперимент" in r for r in out.reasons)


def test_separate_ollama_timeout():
    assert load_settings({**OLLAMA_ENV, "LLM_TIMEOUT_SECONDS": "4"}).timeout_seconds == 4
    assert load_settings({**OLLAMA_ENV, "OLLAMA_TIMEOUT_SECONDS": "15"}).timeout_seconds == 15


def test_missing_model_and_bad_numbers():
    with pytest.raises(LLMConfigError, match="OLLAMA_MODEL"):
        load_settings({"LLM_PROVIDER": "ollama"})
    with pytest.raises(LLMConfigError, match="целыми"):
        load_settings({**OLLAMA_ENV, "OLLAMA_NUM_CTX": "много"})


# ---------- Переключение провайдеров ----------

def test_switching_providers_by_env_only(monkeypatch):
    monkeypatch.setattr(llm_providers.httpx, "get", lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("x")))
    llm.init(OLLAMA_ENV)
    assert isinstance(llm._provider, llm_providers.OllamaProvider)
    llm.init({"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-3.1-flash-lite", "GEMINI_API_KEY": "AIzaFAKE"},
             synthetic_data=True)
    assert isinstance(llm._provider, llm_providers.GeminiProvider)
    llm.init({"LLM_PROVIDER": "disabled"})
    assert llm._provider is None


# ---------- Проверка при старте ----------

def test_startup_check_warns_but_does_not_fail(monkeypatch):
    monkeypatch.setattr(llm_providers.httpx, "get", lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("x")))
    llm.init(OLLAMA_ENV)  # не бросает исключение
    assert "запустите Ollama" in llm.startup_warning()


def test_startup_check_missing_model(monkeypatch):
    tags = {"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b"}]}
    monkeypatch.setattr(llm_providers.httpx, "get",
                        lambda url, **kw: httpx.Response(200, json=tags, request=httpx.Request("GET", url)))
    llm.init(OLLAMA_ENV)
    assert "ollama pull qwen2.5:3b" in llm.startup_warning()


def test_startup_check_ok(ollama):
    assert llm.startup_warning() is None


# ---------- Запрос и успешный ответ ----------

def test_request_format(ollama):
    sent, _ = ollama
    result = _classify()
    assert isinstance(result, llm.LLMCall)
    body = sent[0]["json"]
    assert sent[0]["url"] == f"{URL}/api/chat"
    assert body["model"] == "qwen2.5:3b" and body["stream"] is False
    assert body["format"]["properties"]["category"]["enum"]  # JSON-схема передана в Ollama
    assert body["options"]["num_ctx"] == 4096 and body["options"]["temperature"] == 0
    assert "<player_message>" in body["messages"][1]["content"]


def test_success_tokens_and_zero_cost(ollama):
    result = _classify()
    assert result.result.category == "complaint.insult"
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (1800, 90, 0.0)


# ---------- Ошибки ----------

def test_not_running(ollama):
    _, outcomes = ollama
    outcomes.extend([httpx.ConnectError("refused"), httpx.ConnectError("refused")])
    result = _classify()
    assert result.kind == "network" and "запущена" in result.reason


def test_timeout(ollama):
    _, outcomes = ollama
    outcomes.extend([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])
    assert _classify().kind == "timeout"


def test_model_not_pulled_is_not_retried(ollama):
    sent, outcomes = ollama
    outcomes.append(httpx.Response(404, json={"error": "model 'qwen2.5:3b' not found"},
                                   request=httpx.Request("POST", URL)))
    result = _classify()
    assert result.kind == "not_found" and "ollama pull" in result.reason
    assert len(sent) == 1


def test_server_error_retried_once(ollama):
    sent, outcomes = ollama
    outcomes.append(httpx.Response(500, json={"error": "out of memory"}, request=httpx.Request("POST", URL)))
    assert isinstance(_classify(), llm.LLMCall)
    assert len(sent) == 2


def test_non_json_body(ollama):
    _, outcomes = ollama
    outcomes.append(httpx.Response(200, text="<html>proxy error</html>", request=httpx.Request("POST", URL)))
    assert _classify().kind == "invalid"


def test_content_not_matching_schema(ollama):
    _, outcomes = ollama
    outcomes.append(ollama_reply(_dumps({**VALID, "category": "complaint.made_up"})))
    assert _classify().kind == "invalid"


def test_content_not_json(ollama):
    _, outcomes = ollama
    outcomes.append(ollama_reply("Конечно! Вот ответ: категория оскорбление"))
    assert _classify().kind == "invalid"


def test_empty_content(ollama):
    _, outcomes = ollama
    outcomes.append(ollama_reply(""))
    assert _classify().kind == "empty"


def test_truncated_answer(ollama):
    _, outcomes = ollama
    outcomes.append(ollama_reply('{"category": "complaint.insult", "prio', done_reason="length"))
    assert _classify().kind == "invalid"


def test_unexpected_response_shape(ollama):
    _, outcomes = ollama
    outcomes.append(httpx.Response(200, json=["not", "an", "object"], request=httpx.Request("POST", URL)))
    assert _classify().kind == "invalid"


# ---------- Безопасность не зависит от провайдера ----------

def _request(text, kind="complaint"):
    return TriageRequest(kind=kind, messages=[{"role": "player", "text": text, "attachments": 1}])


def test_ollama_cannot_skip_mandatory_escalation(ollama):
    _, outcomes = ollama
    lenient = {**VALID, "category": "report.question", "priority": "p3", "cited": [], "needs_human": False,
               "confidence": 0.95, "reply": "Ответ"}
    outcomes.extend([ollama_reply(_dumps(lenient))] * 3)
    for text in ("Оплатил донат картой вчера, но ГК так и не пришли на аккаунт",
                 "Мой аккаунт взломали этой ночью, пропали все деньги и машина",
                 "Прошу пересмотреть мой бан, считаю его ошибочным и несправедливым"):
        assert pipeline.triage(_request(text, kind="report")).needs_human is True, text


def test_ollama_promises_are_replaced(ollama):
    _, outcomes = ollama
    outcomes.append(ollama_reply(_dumps({**VALID, "reply": "Не переживайте, мы вас разбаним и вернём деньги"})))
    out = pipeline.triage(_request("Этот игрок долго писал гадости про мою семью в общий чат"))
    assert "разбан" not in out.draft_reply.text and out.needs_human is True


def test_ollama_invented_citation_removed(ollama):
    _, outcomes = ollama
    outcomes.append(ollama_reply(_dumps({**VALID, "cited": ["2.1", "99.9"]})))
    out = pipeline.triage(_request("Этот игрок долго оскорблял мою семью в общем чате, 2.1 ОП"))
    assert out.draft_reply.cited == ["2.1"]


def test_ollama_down_falls_back_to_keywords(ollama):
    _, outcomes = ollama
    outcomes.extend([httpx.ConnectError("refused")] * 2)
    out = pipeline.triage(_request("Этот игрок долго оскорблял мою семью в общем чате, 2.1 ОП"))
    assert out.category == "complaint.insult" and out.needs_human is True
    assert any("ключевым словам" in r for r in out.reasons)
