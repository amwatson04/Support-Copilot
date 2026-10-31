"""Спам и дубликаты.

Спам — сообщение без содержания: пустое, «+», «.», «53», «ааааа»,
и при этом без вложений и ссылок. «помогите» спамом НЕ считаем: это просьба о помощи.

Дубликат — дословный повтор: тот же заявитель про того же игрока
отправил тот же текст в течение 24 часов.
Анализ train показал, что более мягкие правила (просто «та же пара за час»)
совпадают с вердиктом админа лишь в ~13% случаев, поэтому их не используем.
"""
import re
import threading
from datetime import datetime, timedelta, timezone

from app.services.extractor import extract_rule_refs, has_evidence

_TOKENS = re.compile(r"<\w+>")


def is_spam(text: str, attachments: int) -> bool:
    if has_evidence(text, attachments):
        return False
    cleaned = _TOKENS.sub(" ", text).lower()
    letters = [ch for ch in cleaned if ch.isalpha()]
    if not letters:
        return not extract_rule_refs(text)  # «2.1» — осмысленно, «53» или «+» — нет
    return len(set(letters)) == 1 and len(letters) >= 3  # «ааааа», «ххх»


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w<>]+", " ", text.lower()).split())


def _to_utc_naive(moment: datetime | None) -> datetime:
    if moment is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


class DuplicateDetector:
    """Хранит недавние обращения в памяти процесса.

    После перезапуска сервера память очищается. Для хакатона этого достаточно,
    в продакшене здесь была бы база данных.
    """

    def __init__(self, window_hours: int = 24):
        self.window = timedelta(hours=window_hours)
        self._seen: dict[tuple[str, str], list[tuple[datetime, str]]] = {}
        self._last_key: tuple[str, str] | None = None
        self._lock = threading.Lock()  # FastAPI может обрабатывать запросы параллельно

    def check_and_remember(
        self,
        reporter_id: str | None,
        target: str | None,
        text: str,
        created_at: datetime | None = None,
    ) -> bool:
        """True, если это повтор. Обращение запоминается в любом случае."""
        self._last_key = None
        if not reporter_id or not target:
            return False  # не знаем, кто и на кого — сравнивать не с чем
        key = (reporter_id, target.lower())
        moment = _to_utc_naive(created_at)
        normalized = normalize(text)
        if not normalized:
            return False

        with self._lock:
            history = [(t, n) for t, n in self._seen.get(key, []) if abs(moment - t) <= self.window]
            is_duplicate = any(n == normalized for _, n in history)
            history.append((moment, normalized))
            self._seen[key] = history
            self._last_key = key
        return is_duplicate

    def forget_last(self) -> None:
        """Убирает последнее запомненное обращение. Нужно predict.py, когда тикет
        обрабатывается повторно после 429, — иначе он стал бы дубликатом самого себя."""
        with self._lock:
            if self._last_key and self._seen.get(self._last_key):
                self._seen[self._last_key].pop()
            self._last_key = None
