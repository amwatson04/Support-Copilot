"""Провайдеры LLM. Каждый умеет одно: отправить промпт и вернуть JSON-текст.

Ошибки провайдер переводит в ProviderError с понятным видом (kind),
чтобы llm.py решал, повторять запрос или нет, не зная деталей SDK.
Сам провайдер ничего не повторяет и никуда не переключается.
"""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

import httpx

from app.services.llm_config import LLMSettings, redact


@dataclass
class ProviderReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    """kind: quota | forbidden | bad_request | not_found | server | timeout | network | empty | invalid"""

    def __init__(self, kind: str, code: int | None, message: str):
        super().__init__(f"{kind} ({code}): {message}")
        self.kind = kind
        self.code = code
        self.message = message


def _kind_for_status(code: int) -> str:
    if code == 429:
        return "quota"
    if code in (401, 403):
        return "forbidden"
    if code == 404:
        return "not_found"
    if code >= 500:
        return "server"
    return "bad_request"  # 400: в т.ч. неверный ключ (API_KEY_INVALID) и неподдерживаемые параметры


class GeminiProvider:
    # Google отклоняет запросы с таймаутом меньше 10 с (HTTP 400 «Minimum allowed deadline is 10s»)
    MIN_SERVER_TIMEOUT_S = 10

    def __init__(self, settings: LLMSettings):
        from google import genai
        from google.genai import types

        self._settings = settings
        self._types = types
        self._client = genai.Client(
            api_key=settings.api_key,
            http_options=types.HttpOptions(
                timeout=int(max(settings.timeout_seconds, self.MIN_SERVER_TIMEOUT_S) * 1000),  # в миллисекундах
                retry_options=types.HttpRetryOptions(attempts=1),  # повторами управляем сами
            ),
        )
        # Наш собственный лимит ожидания (LLM_TIMEOUT_SECONDS) меньше серверного:
        # запрос идёт в отдельном потоке, и мы перестаём его ждать вовремя.
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini")

    def generate(self, system: str, user: str, schema: dict, timeout_s: float | None = None) -> ProviderReply:
        """timeout_s — сколько ждать этот запрос (llm.py передаёт остаток общего бюджета времени)."""
        from google.genai import errors

        wait = timeout_s or self._settings.timeout_seconds

        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=1024,
            temperature=0,
        )
        # Минимум «размышлений»: быстрее и меньше токенов.
        # Gemini 3 понимает только thinking_level (thinking_budget=0 даёт HTTP 400), Gemini 2.5 — только budget.
        if self._settings.model.startswith("gemini-3"):
            config.thinking_config = types.ThinkingConfig(thinking_level="minimal")
        elif self._settings.model.startswith("gemini-2.5-flash"):
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)

        future = self._executor.submit(
            self._client.models.generate_content, model=self._settings.model, contents=user, config=config
        )
        try:
            response = future.result(timeout=wait)
        except FutureTimeout:
            raise ProviderError("timeout", None, f"нет ответа за {wait:.1f} c") from None
        except errors.APIError as error:
            message = redact(str(error.message or error.status or ""), self._settings.api_key)
            raise ProviderError(_kind_for_status(error.code or 0), error.code, message) from None
        except Exception as error:  # сетевые ошибки httpx/httpx2: таймаут, обрыв соединения
            kind = "timeout" if "Timeout" in type(error).__name__ else "network"
            raise ProviderError(kind, None, type(error).__name__) from None

        text = response.text or ""
        if not text:
            raise ProviderError("empty", None, "пустой ответ модели (возможно, сработал фильтр безопасности)")
        usage = response.usage_metadata
        return ProviderReply(
            text=text,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        )


class OllamaProvider:
    """Локальная LLM через HTTP API Ollama. Данные игроков не покидают компьютер.

    Нужен запущенный Ollama (https://ollama.com) и заранее скачанная модель: `ollama pull <модель>`.
    Модель НЕ скачивается автоматически. Ответ ограничивается JSON-схемой через параметр `format`
    (structured outputs Ollama), затем всё равно проходит ту же проверку LLMResult, что и Gemini.
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings

    def check(self) -> str | None:
        """Проверка при старте: запущен ли Ollama и скачана ли модель.
        Ничего не скачивает и не падает — возвращает текст предупреждения или None."""
        url = self._settings.ollama_url
        try:
            response = httpx.get(f"{url}/api/tags", timeout=2)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return f"Ollama недоступна по адресу {url}: запустите Ollama"
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        model = self._settings.model
        if model not in names and f"{model}:latest" not in names:
            return f"Модель {model} не скачана в Ollama: выполните `ollama pull {model}`"
        return None

    def generate(self, system: str, user: str, schema: dict, timeout_s: float | None = None) -> ProviderReply:
        settings = self._settings
        payload = {
            "model": settings.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,  # держать модель в памяти: повторная загрузка — секунды
            "options": {
                "temperature": 0,
                # По умолчанию Ollama даёт 2048 токенов контекста, а наш промпт ~2100: без этого
                # начало промпта (категории и правила) молча обрезалось бы
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": settings.ollama_num_predict,
            },
        }
        try:
            response = httpx.post(f"{settings.ollama_url}/api/chat", json=payload,
                                  timeout=timeout_s or settings.timeout_seconds)
        except httpx.TimeoutException:
            raise ProviderError("timeout", None, "Ollama не ответила вовремя") from None
        except httpx.ConnectError:
            raise ProviderError("network", None, f"Ollama не запущена ({settings.ollama_url})") from None
        except httpx.HTTPError as error:
            raise ProviderError("network", None, f"Ollama недоступна: {type(error).__name__}") from None

        try:
            data = response.json()
        except ValueError:
            data = None
        error_text = data.get("error") if isinstance(data, dict) else None
        if response.status_code == 404:
            # Ollama отвечает 404, если модель не скачана
            raise ProviderError("not_found", 404, error_text or f"модель {settings.model} не найдена: "
                                                              f"выполните `ollama pull {settings.model}`")
        if response.status_code != 200:
            raise ProviderError(_kind_for_status(response.status_code), response.status_code,
                                (error_text or response.text)[:200])
        if not isinstance(data, dict):
            raise ProviderError("invalid", None, "ответ Ollama не является JSON-объектом")
        if error_text:
            raise ProviderError("bad_request", None, str(error_text)[:200])

        message = data.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if data.get("done_reason") == "length":
            # Ответ упёрся в OLLAMA_NUM_PREDICT — JSON обрезан и не пройдёт проверку
            raise ProviderError("invalid", None, "ответ обрезан по длине: увеличьте OLLAMA_NUM_PREDICT")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("empty", None, "пустой ответ Ollama")
        return ProviderReply(text, int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0))
