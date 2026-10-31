"""Точка входа FastAPI. Здесь только HTTP, вся логика — в app/pipeline.py."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from app.pipeline import triage as run_triage
from app.schemas import TriageRequest, TriageResponse
from app.services import llm
from app.services.rag import TAXONOMY_PATH
from app.services.rules import RULES_PATH, get_rule


def missing_data_files() -> list[str]:
    """Файлы кейса, без которых сервис не работает (их нет в git — условия кейса)."""
    return [f"data/{p.name}" for p in (RULES_PATH, TAXONOMY_PATH) if not p.exists()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Без файлов кейса /triage падал бы с 500 при первом же обращении — останавливаем запуск сразу
    missing = missing_data_files()
    if missing:
        raise RuntimeError(f"Нет файлов кейса: {', '.join(missing)}. Положите их в папку data/ (см. README)")
    # Ошибка в .env (запрещённая модель, нет ключа) останавливает запуск с понятным сообщением
    llm.init()
    yield


app = FastAPI(title="Support Copilot by Grand Mobile", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    settings = llm.current_settings()
    result = {"status": "ok", "llm_provider": settings.provider if settings else "not_initialized"}
    if settings and settings.model:
        result["llm_model"] = settings.model
    if llm.startup_warning():
        result["llm_warning"] = llm.startup_warning()
    return result


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> TriageResponse:
    return run_triage(request)


@app.get("/rules")
def rules(ids: list[str] = Query(default=[])) -> list[dict]:
    """Текст пунктов правил по ID, например /rules?ids=2.1&ids=GZ%201.2. Несуществующие ID пропускаются."""
    found = [get_rule(rule_id) for rule_id in ids]
    return [{"id": r.id, "document": r.document, "text": r.text} for r in found if r]
