"""Модели запроса и ответа POST /triage.

Это контракт с frontend: названия полей не меняем без согласования.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["complaint", "report"]
Lang = Literal["ru", "kk", "uz", "en", "es", "pt"]
Priority = Literal["p0", "p1", "p2", "p3"]
Category = Literal[
    "complaint.insult",
    "complaint.cheat",
    "complaint.dm",
    "complaint.scam",
    "complaint.rmt_ads",
    "complaint.admin",
    "complaint.other",
    "complaint.invalid",
    "report.question",
    "report.help",
    "report.complaint",
    "report.account",
    "report.payment",
    "report.offtopic",
]


# ---------- Запрос ----------

class Message(BaseModel):
    role: Literal["player", "reported_player", "admin"] = "player"
    text: str = Field(default="", max_length=5000)
    attachments: int = Field(default=0, ge=0, le=100)


class TriageRequest(BaseModel):
    kind: Kind
    messages: list[Message] = Field(min_length=1, max_length=50)

    # Необязательные поля. Frontend может их не передавать.
    # Нужны для дедупликации и выбора языка, когда текст пустой («+»).
    # Лимиты длины — защита от слишком больших запросов.
    reporter_id: str | None = Field(default=None, max_length=128)    # кто подал (reporter_hash в датасете)
    reported_nick: str | None = Field(default=None, max_length=64)   # на кого жалоба
    created_at: datetime | None = None                               # когда подано; по умолчанию — сейчас
    server_lang: str | None = Field(default=None, max_length=8)      # язык сервера: ru | kk | uz ...


# ---------- Ответ ----------

class Extracted(BaseModel):
    nickname: str | None = None
    rule_refs: list[str] = []
    has_evidence: bool = False
    server: str | None = None


class DraftReply(BaseModel):
    lang: Lang
    text: str
    cited: list[str] = []


class TriageResponse(BaseModel):
    lang: Lang
    category: Category
    priority: Priority
    is_spam_or_duplicate: bool
    extracted: Extracted
    draft_reply: DraftReply
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = []
