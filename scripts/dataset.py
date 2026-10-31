"""Чтение тикетов датасета и превращение их в запрос /triage."""
import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from app.schemas import TriageRequest


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def request_from_record(record: dict) -> TriageRequest:
    """Ответ админа в запрос не попадает: его модель видеть не должна."""
    messages = []
    for message in record["messages"]:
        if message["role"] == "admin":
            continue
        # В датасете скриншоты посчитаны на весь тикет — приписываем их первому сообщению игрока
        is_first_player = message["role"] == "player" and not any(m["role"] == "player" for m in messages)
        messages.append({
            "role": message["role"],
            "text": message.get("text") or "",
            "attachments": (record.get("images_count") or 0) if is_first_player else 0,
        })
    if not messages:
        messages = [{"role": "player", "text": "", "attachments": 0}]
    return TriageRequest(
        kind=record["kind"],
        messages=messages,
        reporter_id=record.get("reporter_hash"),
        reported_nick=record.get("reported_nick"),
        created_at=record.get("created_at"),
        server_lang=record.get("server_lang"),
    )
