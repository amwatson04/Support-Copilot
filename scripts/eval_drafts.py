"""Черновики и задержка на синтетических обращениях (scripts/synthetic_tickets.py) — одинаковые условия для провайдеров.

Запуск (провайдер должен быть доступен):
    python -m scripts.eval_drafts --provider gemini
    OLLAMA_MODEL=qwen2.5:3b python -m scripts.eval_drafts --provider ollama     # Ollama запущена отдельно
    python -m scripts.eval_drafts --blind      # перемешать черновики всех провайдеров для слепой оценки

Проход 1: первый запрос — «холодный» (Ollama грузит модель в память, у Gemini — новое соединение).
Проход 2: те же обращения ещё раз с очищенным кэшем — «прогретые» запросы, по ним p50/p95.
Таймаут намеренно большой (60 с): меряется скорость модели, а не обрезка по LLM_TIMEOUT_SECONDS.
"""
import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

from app import pipeline
from app.schemas import TriageRequest
from app.services import llm
from scripts.synthetic_tickets import SYNTHETIC

OUT = Path("outputs")


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def run(provider: str) -> None:
    env = dict(os.environ)
    env.update({"LLM_PROVIDER": provider, "LLM_TIMEOUT_SECONDS": "60", "OLLAMA_TIMEOUT_SECONDS": "60",
                "LLM_MAX_RETRIES": "0"})
    from dotenv import dotenv_values

    from app.services.llm_config import ENV_PATH
    for key, value in dotenv_values(ENV_PATH).items():  # ключ и модель из .env, если не заданы в окружении
        env.setdefault(key, value or "")
    settings = llm.init(env, synthetic_data=True)  # только тикеты из scripts/synthetic_tickets.py
    llm.set_rate_limit(12 if provider == "gemini" else None)
    print(f"Провайдер: {settings.provider} {settings.model}")

    raw: dict[str, str] = {}
    original = llm.classify

    def capture(**kwargs):
        call = original(**kwargs)
        if isinstance(call, llm.LLMCall):
            raw[kwargs["text"]] = call.result.reply
        return call
    llm.classify = capture

    rows, latencies = [], {1: [], 2: []}
    for run_no in (1, 2):
        llm._cache.clear()
        for sid, kind, text, attachments, expected in SYNTHETIC:
            pipeline.duplicates.__init__()
            request = TriageRequest(kind=kind, messages=[{"role": "player", "text": text, "attachments": attachments}])
            started = time.perf_counter()
            out = pipeline.triage(request)
            elapsed = time.perf_counter() - started - llm.last_rate_wait_s
            llm.last_rate_wait_s = 0.0
            used_llm = any(r.startswith("Пояснение LLM") for r in out.reasons) or text in raw
            latencies[run_no].append(elapsed)
            if run_no == 1:
                rows.append({"id": sid, "provider": f"{settings.provider}:{settings.model}", "kind": kind,
                             "text": text, "expected": expected, "lang": out.lang, "category": out.category,
                             "needs_human": out.needs_human, "cited": out.draft_reply.cited,
                             "raw_draft": raw.get(text), "final_draft": out.draft_reply.text,
                             "used_llm": used_llm, "reasons": out.reasons, "latency_s": round(elapsed, 2)})
    OUT.mkdir(exist_ok=True)
    path = OUT / f"drafts_{settings.provider}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    llm_cold = rows[0]["latency_s"]
    warm = latencies[2]
    print(f"Записано: {path}")
    print(f"Холодный старт (первый запрос): {llm_cold:.2f} c")
    print(f"Прогретые (проход 2, n={len(warm)}): p50 {statistics.median(warm):.2f} c, p95 {percentile(warm, 0.95):.2f} c, "
          f"max {max(warm):.2f} c")
    print(f"Черновиков модели: {sum(r['raw_draft'] is not None for r in rows)}/{len(rows)}; "
          f"заменено шаблоном: {sum(r['raw_draft'] is not None and r['raw_draft'] != r['final_draft'] for r in rows)}")


def blind() -> None:
    items = []
    for path in sorted(OUT.glob("drafts_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["raw_draft"]:
                items.append(row)
    random.Random(2026).shuffle(items)
    key, sheet = {}, []
    for i, row in enumerate(items, 1):
        blind_id = f"B{i:02d}"
        key[blind_id] = row["provider"]
        sheet.append({"blind_id": blind_id, "ticket": row["text"], "expected": row["expected"],
                      "lang": row["lang"], "category": row["category"], "cited": row["cited"],
                      "draft": row["raw_draft"]})
    (OUT / "blind_sheet.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in sheet) + "\n",
                                           encoding="utf-8")
    (OUT / "blind_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Слепой лист: outputs/blind_sheet.jsonl ({len(sheet)} черновиков), ключ: outputs/blind_key.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "ollama"])
    parser.add_argument("--blind", action="store_true")
    args = parser.parse_args()
    blind() if args.blind else run(args.provider)
