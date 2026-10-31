"""Пакетная обработка JSONL: каждый тикет -> ответ /triage, одна строка на тикет.

Запуск из корня проекта:
    python -m scripts.predict                                   # data/tickets_valid.jsonl
    python -m scripts.predict --input data/x.jsonl --output outputs/x.jsonl
    python -m scripts.predict --limit 50                        # первые 50 тикетов
    python -m scripts.predict --provider disabled               # без LLM, быстро

Что умеет:
- выдерживает лимит бесплатного тарифа (--rpm, по умолчанию 12 запросов в минуту);
- при HTTP 429 ждёт и повторяет тот же тикет; если квота не возвращается — останавливается;
- продолжает с места остановки: уже обработанные ticket_id в выходном файле пропускаются;
- каждую строку пишет сразу, поэтому Ctrl+C или ошибка не теряют сделанную работу.
"""
import argparse
import collections
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from app import pipeline
from app.services import llm
from app.services.llm_config import ENV_PATH
from scripts.dataset import read_jsonl, request_from_record

QUOTA_WAITS_S = (60, 120, 240)  # паузы при 429; после последней считаем, что кончилась дневная квота


def done_ticket_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["ticket_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # недописанная строка после аварийной остановки — тикет обработаем заново
    return done


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))] if ordered else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Copilot: обработка JSONL")
    parser.add_argument("--input", default="data/tickets_valid.jsonl")
    parser.add_argument("--output", default="outputs/predictions_valid.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="обработать не больше N новых тикетов")
    parser.add_argument("--rpm", type=float, default=12, help="запросов к LLM в минуту (0 — без ограничения)")
    parser.add_argument("--provider", choices=["disabled", "ollama"],
                        help="переопределить LLM_PROVIDER (Gemini для реальных тикетов запрещён)")
    parser.add_argument("--timeout", type=float, help="переопределить LLM_TIMEOUT_SECONDS (сек. на один ответ LLM)")
    parser.add_argument("--fresh", action="store_true",
                        help="начать заново: очистить выходной файл (иначе готовые тикеты НЕ пересчитываются)")
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    env = dict(os.environ)
    if args.provider:
        env["LLM_PROVIDER"] = args.provider
    if args.timeout:
        env["LLM_TIMEOUT_SECONDS"] = str(args.timeout)
    settings = llm.init(env)
    llm.set_rate_limit(args.rpm if settings.provider == "gemini" else None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and output.exists():
        output.write_text("", encoding="utf-8")
    done = done_ticket_ids(output)
    if done:
        print(f"ВНИМАНИЕ: в {output} уже {len(done)} готовых тикетов — они НЕ пересчитываются (продолжение с места "
              f"остановки). После изменения кода или .env используйте --fresh или другой --output.")

    # По времени подачи: так дедупликация видит обращения в том же порядке, что и в жизни
    records = sorted(read_jsonl(args.input), key=lambda r: r.get("created_at") or "")
    todo = [r for r in records if r["ticket_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Провайдер: {settings.provider} {settings.model}. Всего {len(records)}, уже готово {len(done)}, "
          f"обработаем сейчас {len(todo)}. Результат: {output}")

    # Уже обработанные тикеты прогоняем через детектор дубликатов, чтобы он помнил историю
    for record in records:
        if record["ticket_id"] in done:
            request = request_from_record(record)
            text = "\n".join(m.text for m in request.messages if m.role == "player")
            pipeline.duplicates.check_and_remember(request.reporter_id, request.reported_nick, text, request.created_at)

    # Запоминаем исход последнего обращения к LLM, чтобы отличить 429 от остальных ошибок
    last_call: list = [None]
    original_classify = llm.classify

    def tracking_classify(**kwargs):
        last_call[0] = original_classify(**kwargs)
        return last_call[0]
    llm.classify = tracking_classify

    latencies, categories, outcomes = [], collections.Counter(), collections.Counter()
    total_cost, input_tokens, output_tokens = 0.0, 0, 0
    stopped_by_quota = False
    with output.open("a", encoding="utf-8") as out_file:
        for i, record in enumerate(todo, 1):
            for wait in (*QUOTA_WAITS_S, None):
                last_call[0] = None
                llm.last_rate_wait_s = 0.0
                started = time.perf_counter()
                result = pipeline.triage(request_from_record(record))
                elapsed = time.perf_counter() - started - llm.last_rate_wait_s
                call = last_call[0]
                if not (isinstance(call, llm.LLMFailure) and call.kind == "quota") or wait is None:
                    break
                # Ответ с 429 не сохраняем: подождём и обработаем этот же тикет ещё раз
                pipeline.duplicates.forget_last()
                print(f"  429: квота исчерпана, ждём {wait} c ...")
                time.sleep(wait)

            if isinstance(call, llm.LLMFailure) and call.kind == "quota":
                stopped_by_quota = True
                print("Квота не восстановилась — похоже, исчерпан дневной лимит. "
                      "Запустите ту же команду позже: обработка продолжится с этого тикета.")
                break

            out_file.write(json.dumps({"ticket_id": record["ticket_id"], **result.model_dump()}, ensure_ascii=False) + "\n")
            out_file.flush()

            latencies.append(elapsed)
            categories[result.category] += 1
            if call is None:
                outcomes["без LLM"] += 1
            elif isinstance(call, llm.LLMFailure):
                outcomes[f"ошибка {call.kind}"] += 1
            else:
                outcomes["кэш" if call.cached else "LLM"] += 1
                total_cost += call.cost_usd
                input_tokens += call.input_tokens
                output_tokens += call.output_tokens
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  p95 {percentile(latencies, 0.95):.2f}s  {dict(outcomes)}")

    print("\n=== Итог")
    print(f"Обработано сейчас: {len(latencies)}; всего в файле: {len(done_ticket_ids(output))}/{len(records)}")
    if latencies:
        print(f"Задержка (без пауз ограничителя): p50 {percentile(latencies, 0.5):.2f}s, "
              f"p95 {percentile(latencies, 0.95):.2f}s, среднее {statistics.mean(latencies):.2f}s")
    print(f"Источник ответа: {dict(outcomes)}")
    print(f"Категории: {dict(categories.most_common())}")
    print(f"Токены LLM: вход {input_tokens}, выход {output_tokens}. "
          f"Стоимость: ${total_cost:.4f} (токены × LLM_PRICE_* из .env; бесплатный тариф и без LLM — $0)")
    if stopped_by_quota:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
