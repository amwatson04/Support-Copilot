"""Замер задержки, токенов и стоимости на тикетах из valid.

Запуск (из корня проекта, LLM_PROVIDER включён в .env):
    python -m scripts.measure_llm --n 30
Каждый тикет — максимум один запрос к LLM; одинаковые тексты берутся из кэша.
"""
import argparse
import collections
import random
import statistics
import time

from app import pipeline
from app.services import llm
from scripts.dataset import read_jsonl, request_from_record


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="сколько тикетов обработать")
    parser.add_argument("--file", default="data/tickets_valid.jsonl")
    args = parser.parse_args()

    settings = llm.init()
    if settings.provider == "disabled":
        raise SystemExit("LLM_PROVIDER=disabled в .env — замер LLM невозможен")
    print(f"Провайдер: {settings.provider}, модель: {settings.model}\n")

    # Перехватываем вызовы LLM, чтобы записать результат каждого
    calls: list = []
    original = llm.classify

    def recording_classify(**kwargs):
        call = original(**kwargs)
        calls.append(call)
        return call
    llm.classify = recording_classify

    records = list(read_jsonl(args.file))
    random.seed(42)
    sample = random.sample(records, min(args.n, len(records)))

    latencies = []
    for i, record in enumerate(sample, 1):
        started = time.perf_counter()
        out = pipeline.triage(request_from_record(record))
        latencies.append(time.perf_counter() - started)
        text = " ".join(m["text"] or "" for m in record["messages"] if m["role"] == "player")[:50]
        print(f"{i:3}. {latencies[-1]:5.2f}s  {out.category:18} {out.priority}  human={out.needs_human!s:5}  {text!r}")

    ok = [c for c in calls if isinstance(c, llm.LLMCall)]
    real = [c for c in ok if not c.cached]
    failures = collections.Counter(c.kind for c in calls if isinstance(c, llm.LLMFailure))
    print("\n=== Итог")
    print(f"Тикетов: {len(sample)}, обращений к LLM: {len(calls)} "
          f"(реальных запросов: {len(real)}, из кэша: {len(ok) - len(real)}), ошибок: {dict(failures) or 0}")
    print(f"Задержка пайплайна: p50 {percentile(latencies, 0.5):.2f}s, p95 {percentile(latencies, 0.95):.2f}s, "
          f"max {max(latencies):.2f}s   (лимит p95: 5s)")
    if real:
        print(f"Токены на запрос: вход ~{statistics.mean(c.input_tokens for c in real):.0f}, "
              f"выход ~{statistics.mean(c.output_tokens for c in real):.0f}")
        print(f"Стоимость: ${sum(c.cost_usd for c in real):.4f} "
              "(токены × LLM_PRICE_* из .env; бесплатный тариф и локальная модель — $0)")


if __name__ == "__main__":
    main()
