"""Настройки LLM из .env и защита от случайных расходов.

LLM_PROVIDER:
    disabled — (по умолчанию) облачная LLM не вызывается вообще; данные игроков никуда не уходят.
    gemini   — Google Gemini API, только модели из GEMINI_ALLOWED_MODELS.
    ollama   — локальная модель через Ollama, данные не покидают компьютер.

Чего код НЕ может гарантировать: что к Google-проекту ключа не привязан Billing.
Это проверяет человек в Google AI Studio / Cloud Console.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# .env читается строго из корня проекта: load_dotenv() без пути ищет файл вверх по папкам
# и может подхватить чужой .env (с чужим ключом) из родительского каталога
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

PROVIDERS = ("disabled", "gemini", "ollama")

# Модели с бесплатным тарифом по странице https://ai.google.dev/gemini-api/docs/pricing (проверено 2026-10-30).
# Добавлять модель сюда — только после проверки, что у неё есть Free Tier.
# Замер 2026-10-30: gemini-3.1-flash-lite отвечает ~1 c, gemini-3.5-flash-lite — 4-6 c (не укладывается в p95 5 c),
# модели 2.5 возвращают 404 «no longer available to new users».
GEMINI_ALLOWED_MODELS = frozenset({
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
})


ALL_LANGS = ("ru", "kk", "uz", "en", "es", "pt")

# Ключ, вставленный не в ту строку .env (GEMINI_MODEL=AIza…), не должен попасть в текст ошибки,
# лог или /health. Ключи Google: «AIza…» (39 символов) или «AQ.…»; имена моделей короче и с точками/двоеточиями
_SECRET_LIKE = re.compile(r"AIza[\w\-]{10,}|AQ\.[\w.\-]{10,}|[A-Za-z0-9_\-]{32,}")


def _shown(value: str) -> str:
    """Значение для сообщения об ошибке: как есть, если не похоже на секрет."""
    return "<скрыто: похоже на API-ключ — проверьте эту строку в .env>" if _SECRET_LIKE.search(value) else repr(value)


def _draft_langs(env, default: tuple[str, ...]) -> frozenset:
    """LLM_DRAFT_LANGS=ru,en,uz — для каких языков использовать черновик модели. Пусто — значение по умолчанию."""
    raw = (env.get("LLM_DRAFT_LANGS") or "").strip()
    if not raw:
        return frozenset(default)
    langs = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = sorted(part.strip() for part in raw.split(",") if part.strip().lower() in langs - set(ALL_LANGS))
    if unknown:
        raise LLMConfigError(f"LLM_DRAFT_LANGS: неизвестные языки {_shown(', '.join(unknown))}. Допустимо: {', '.join(ALL_LANGS)}")
    return frozenset(langs)


class LLMConfigError(Exception):
    """Ошибка конфигурации. Приложение с ней не запускается."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str = ""
    api_key: str = ""          # никогда не логируем
    timeout_seconds: float = 4.0
    max_retries: int = 1       # повторы только при сетевых ошибках и 5xx
    ollama_url: str = "http://localhost:11434"
    # $ за 1 млн токенов. 0 — бесплатный тариф / локальная модель. Нужны только для оценки стоимости.
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    # Только для Ollama
    ollama_num_ctx: int = 4096        # окно контекста; по умолчанию у Ollama 2048 — меньше нашего промпта
    ollama_num_predict: int = 512     # максимум токенов ответа
    ollama_keep_alive: str = "10m"    # сколько держать модель в памяти после запроса
    # Языки, для которых черновику модели доверяем; для остальных — ручной шаблон на языке игрока
    draft_langs: frozenset = frozenset(ALL_LANGS)

    def __repr__(self) -> str:  # чтобы ключ не попал в лог даже при print(settings)
        return f"LLMSettings(provider={self.provider!r}, model={self.model!r}, api_key=***)"


def load_settings(env: dict | None = None) -> LLMSettings:
    env = os.environ if env is None else env
    provider = (env.get("LLM_PROVIDER") or "disabled").strip().lower()
    if provider not in PROVIDERS:
        raise LLMConfigError(f"LLM_PROVIDER={_shown(provider)} не поддерживается. Допустимо: {', '.join(PROVIDERS)}")

    timeout = float(env.get("LLM_TIMEOUT_SECONDS") or 4)
    retries = int(env.get("LLM_MAX_RETRIES") or 1)
    prices = {"price_input_per_m": float(env.get("LLM_PRICE_INPUT_PER_M") or 0),
              "price_output_per_m": float(env.get("LLM_PRICE_OUTPUT_PER_M") or 0)}

    if provider == "disabled":
        return LLMSettings(provider="disabled")

    if provider == "gemini":
        model = (env.get("GEMINI_MODEL") or "").strip()
        if model not in GEMINI_ALLOWED_MODELS:
            raise LLMConfigError(
                f"GEMINI_MODEL={_shown(model)} не входит в список разрешённых бесплатных моделей: "
                f"{', '.join(sorted(GEMINI_ALLOWED_MODELS))}"
            )
        api_key = (env.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise LLMConfigError("LLM_PROVIDER=gemini, но GEMINI_API_KEY в .env пуст")
        return LLMSettings("gemini", model, api_key, timeout, retries, **prices,
                           draft_langs=_draft_langs(env, default=ALL_LANGS))

    # ollama: модель задаётся явно, автоматически ничего не скачивается
    model = (env.get("OLLAMA_MODEL") or "").strip()
    if not model:
        raise LLMConfigError("LLM_PROVIDER=ollama, но OLLAMA_MODEL в .env пуст (например, qwen2.5:3b)")
    if _SECRET_LIKE.search(model):  # иначе имя модели попадёт в лог, /health и интерфейс
        raise LLMConfigError(f"OLLAMA_MODEL={_shown(model)} — ожидается имя модели, например qwen2.5:3b")
    url = (env.get("OLLAMA_BASE_URL") or env.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
    # Ollama — «локальная» модель: удалённый адрес означал бы отправку реальных тикетов на чужой сервер
    host = urlparse(url).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise LLMConfigError(f"OLLAMA_BASE_URL должен указывать на этот компьютер (localhost), а не на {_shown(host)}")
    # Локальная модель на ноутбуке медленнее облака — свой таймаут, если задан
    ollama_timeout = float(env.get("OLLAMA_TIMEOUT_SECONDS") or timeout)
    try:
        num_ctx = int(env.get("OLLAMA_NUM_CTX") or 4096)
        num_predict = int(env.get("OLLAMA_NUM_PREDICT") or 512)
    except ValueError:
        raise LLMConfigError("OLLAMA_NUM_CTX и OLLAMA_NUM_PREDICT должны быть целыми числами") from None
    return LLMSettings(
        "ollama", model, "", ollama_timeout, retries, url, **prices,
        ollama_num_ctx=num_ctx, ollama_num_predict=num_predict,
        ollama_keep_alive=(env.get("OLLAMA_KEEP_ALIVE") or "10m").strip(),
        # Слепая оценка qwen2.5:3b (2026-10-31): свои черновики модели — 2.94 из 5 против 3.74 у шаблонов,
        # 10 из 31 с выдуманными фактами или обвинением заявителя; kk/uz бессвязные. По умолчанию — только шаблоны,
        # модель классифицирует. Включить черновики модели: LLM_DRAFT_LANGS=ru,en
        draft_langs=_draft_langs(env, default=()),
    )


def redact(text: str, secret: str) -> str:
    """Убирает секрет из строки перед записью в лог."""
    return text.replace(secret, "***") if secret else text
