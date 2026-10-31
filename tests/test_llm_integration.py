"""Безопасность интеграции LLM. Настоящий Gemini не вызывается: подменяем метод SDK."""
import json
import logging

import httpx
import pytest
from google.genai import errors

from app.services import llm
from app.services.llm_config import GEMINI_ALLOWED_MODELS, LLMConfigError, load_settings
from app.services.rules import get_rule

FAKE_KEY = "AIzaFAKE-test-key-1234567890"
MODEL = "gemini-3.1-flash-lite"
GOOD_ENV = {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": MODEL, "GEMINI_API_KEY": FAKE_KEY}
VALID_JSON = json.dumps({"category": "complaint.insult", "priority": "p2", "cited": ["2.1"], "reply": "Черновик",
                         "needs_human": True, "confidence": 0.8, "reason": "Оскорбление"})


# ---------- Конфигурация ----------

def test_allowed_model():
    settings = load_settings(GOOD_ENV)
    assert settings.provider == "gemini" and settings.model in GEMINI_ALLOWED_MODELS


def test_forbidden_model_stops_start():
    with pytest.raises(LLMConfigError, match="не входит в список"):
        load_settings({**GOOD_ENV, "GEMINI_MODEL": "gemini-3.1-pro-preview"})


def test_missing_api_key():
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        load_settings({**GOOD_ENV, "GEMINI_API_KEY": ""})


def test_unknown_provider_rejected():
    with pytest.raises(LLMConfigError):
        load_settings({"LLM_PROVIDER": "openai"})


def test_disabled_by_default():
    assert load_settings({}).provider == "disabled"
    llm.init({})
    assert isinstance(_classify(), llm.LLMFailure)
    assert _classify().kind == "disabled"


def test_key_hidden_in_repr():
    assert FAKE_KEY not in repr(load_settings(GOOD_ENV))


# ---------- Вызовы с подменённым SDK ----------

class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None


def _classify():
    return llm.classify(kind="complaint", text="оск родни", lang="ru", attachments=1,
                        player_refs=[], rules=[get_rule("2.1")])


@pytest.fixture
def gemini(monkeypatch):
    """Включает Gemini с фейковым ключом и записывает все обращения к SDK."""
    llm.init(GOOD_ENV, synthetic_data=True)
    calls = []
    outcomes = []

    def fake_generate_content(model, contents, config):
        calls.append(model)
        outcome = outcomes.pop(0) if outcomes else FakeResponse(VALID_JSON)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm._provider._client.models, "generate_content", fake_generate_content)
    return calls, outcomes


def test_successful_call(gemini):
    calls, _ = gemini
    result = _classify()
    assert isinstance(result, llm.LLMCall)
    assert result.result.category == "complaint.insult"
    assert result.cost_usd == 0.0
    assert calls == [MODEL]


def test_cache_avoids_second_call(gemini):
    calls, _ = gemini
    _classify()
    second = _classify()
    assert second.cached is True and len(calls) == 1


@pytest.mark.parametrize("code, kind", [(429, "quota"), (403, "forbidden"), (400, "bad_request"), (404, "not_found")])
def test_client_errors_not_retried_and_no_fallback(gemini, code, kind):
    calls, outcomes = gemini
    outcomes.append(errors.ClientError(code, {"error": {"code": code, "message": "x", "status": "X"}}))
    result = _classify()
    assert isinstance(result, llm.LLMFailure) and result.kind == kind
    assert calls == [MODEL]  # ровно один вызов, никакой другой модели


def test_server_error_retried_once_same_model(gemini):
    calls, outcomes = gemini
    outcomes.append(errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}))
    result = _classify()
    assert isinstance(result, llm.LLMCall)
    assert calls == [MODEL, MODEL]


def test_network_error(gemini):
    _, outcomes = gemini
    outcomes.extend([httpx.ConnectError("no route"), httpx.ConnectError("no route")])
    result = _classify()
    assert isinstance(result, llm.LLMFailure) and result.kind == "network"


def test_timeout(gemini):
    _, outcomes = gemini
    outcomes.extend([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])
    assert _classify().kind == "timeout"


def test_invalid_json_from_model(gemini):
    _, outcomes = gemini
    outcomes.append(FakeResponse('{"category": "made.up"}'))
    assert _classify().kind == "invalid"


def test_key_not_leaked_in_logs(gemini, caplog):
    _, outcomes = gemini
    # Даже если Google вернёт ключ в тексте ошибки, в лог он не попадёт
    outcomes.append(errors.ClientError(400, {"error": {"code": 400, "message": f"API key {FAKE_KEY} not valid",
                                                       "status": "INVALID_ARGUMENT"}}))
    with caplog.at_level(logging.DEBUG):
        result = _classify()
    assert result.kind == "bad_request"
    assert FAKE_KEY not in caplog.text
    assert "***" in caplog.text


def test_key_not_leaked_on_init(caplog):
    with caplog.at_level(logging.DEBUG):
        llm.init(GOOD_ENV, synthetic_data=True)
    assert FAKE_KEY not in caplog.text


def test_key_not_leaked_in_api_response_and_health(gemini):
    from fastapi.testclient import TestClient
    from app.main import app
    _, outcomes = gemini
    outcomes.append(errors.ClientError(400, {"error": {"code": 400, "message": f"API key {FAKE_KEY} not valid",
                                                       "status": "INVALID_ARGUMENT"}}))
    client = TestClient(app)  # без lifespan: настройки уже выставил fixture gemini
    body = client.post("/triage", json={"kind": "complaint", "messages": [
        {"text": "Игрок у банка долго ходил за мной и мешал, разберитесь пожалуйста"}]}).text
    assert FAKE_KEY not in body and "проверьте ключ" in body  # причина понятна, ключа нет
    assert FAKE_KEY not in client.get("/health").text


# Ключ, вставленный не в ту строку .env, не печатается в сообщении об ошибке
@pytest.mark.parametrize("env", [
    {"LLM_PROVIDER": FAKE_KEY + "0123456789"},
    {**GOOD_ENV, "GEMINI_MODEL": FAKE_KEY},
    {**GOOD_ENV, "LLM_DRAFT_LANGS": FAKE_KEY},
    {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": FAKE_KEY},
])
def test_key_in_wrong_env_line_not_echoed(env):
    with pytest.raises(LLMConfigError) as error:
        load_settings(env)
    assert FAKE_KEY.lower() not in str(error.value).lower()
    assert "похоже на API-ключ" in str(error.value)


def test_normal_config_errors_still_show_value():
    with pytest.raises(LLMConfigError, match="'gemini,'"):  # опечатку видно, чтобы её можно было исправить
        load_settings({"LLM_PROVIDER": "gemini,"})
    assert load_settings({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5:3b"}).model == "qwen2.5:3b"


# ---------- Замечания ревью ----------

def test_cache_respects_exact_attachment_count(gemini):
    calls, _ = gemini
    for count in (1, 3, 1):
        llm.classify(kind="complaint", text="оск родни", lang="ru", attachments=count,
                     player_refs=[], rules=[get_rule("2.1")])
    assert len(calls) == 2  # 1 и 3 скриншота — разные промпты; повтор с 1 — из кэша


def test_confidence_out_of_range_is_clamped(gemini):
    _, outcomes = gemini
    outcomes.append(FakeResponse(VALID_JSON.replace('"confidence": 0.8', '"confidence": 1.7')))
    result = _classify()
    assert isinstance(result, llm.LLMCall) and result.result.confidence == 1.0


def test_too_long_reply_is_rejected(gemini):
    _, outcomes = gemini
    too_long = {**json.loads(VALID_JSON), "reply": "я" * (llm.MAX_REPLY_CHARS + 1)}
    outcomes.append(FakeResponse(json.dumps(too_long)))
    assert _classify().kind == "invalid"


def test_retries_fit_total_time_budget(monkeypatch):
    """Повтор не растягивает ответ: все попытки укладываются в LLM_TIMEOUT_SECONDS."""
    import time
    llm.init({**GOOD_ENV, "LLM_TIMEOUT_SECONDS": "1.5", "LLM_MAX_RETRIES": "3"}, synthetic_data=True)
    calls = []

    def slow(model, contents, config):
        calls.append(model)
        time.sleep(2)  # дольше бюджета
        return FakeResponse(VALID_JSON)

    monkeypatch.setattr(llm._provider._client.models, "generate_content", slow)
    started = time.perf_counter()
    result = _classify()
    elapsed = time.perf_counter() - started
    assert result.kind == "timeout"
    assert elapsed < 1.9          # ждали не дольше бюджета, а не 1.5 c × 4 попытки
    assert len(calls) == 1        # на повтор времени не осталось


def test_cost_is_computed_from_tokens_and_prices(monkeypatch):
    llm.init({**GOOD_ENV, "LLM_PRICE_INPUT_PER_M": "2", "LLM_PRICE_OUTPUT_PER_M": "10"}, synthetic_data=True)

    class Usage:
        prompt_token_count, candidates_token_count, thoughts_token_count = 1000, 100, 0

    response = FakeResponse(VALID_JSON)
    response.usage_metadata = Usage()
    monkeypatch.setattr(llm._provider._client.models, "generate_content", lambda **kw: response)
    result = _classify()
    assert result.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1_000_000)
    assert _classify().cost_usd == 0.0  # из кэша — запроса не было


def test_free_tier_cost_is_zero(gemini):
    assert _classify().cost_usd == 0.0


# ---------- Gemini Free Tier запрещён для реальных тикетов ----------

def test_gemini_blocked_without_synthetic_flag():
    with pytest.raises(LLMConfigError, match="запрещён для реальных тикетов"):
        llm.init(GOOD_ENV)
    llm.init(GOOD_ENV, synthetic_data=True)  # синтетические эксперименты (scripts.eval_drafts) — можно
    assert llm.current_settings().provider == "gemini"


def test_api_does_not_start_with_gemini(monkeypatch):
    # API и Streamlit (ходит только через API): с LLM_PROVIDER=gemini сервер не запускается
    from fastapi.testclient import TestClient
    from app.main import app
    for name, value in GOOD_ENV.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(LLMConfigError, match="запрещён для реальных тикетов"):
        with TestClient(app):
            pass


def test_batch_refuses_gemini(monkeypatch, tmp_path):
    import sys
    from scripts import predict
    for name, value in GOOD_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["predict", "--input", str(tmp_path / "in.jsonl"),
                                      "--output", str(tmp_path / "out.jsonl")])
    with pytest.raises(LLMConfigError, match="запрещён для реальных тикетов"):
        predict.main()
    monkeypatch.setattr(sys, "argv", ["predict", "--provider", "gemini"])
    with pytest.raises(SystemExit):  # argparse: такого значения нет
        predict.main()
