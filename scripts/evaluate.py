"""Метрики качества по файлу предсказаний predict.py.

Запуск:
    python -m scripts.evaluate --pred outputs/predictions_valid.jsonl

Три уровня проверки:
1. Инварианты безопасности — должны быть 100% на ЛЮБОМ прогоне
   (цитаты существуют, нет запрещённых обещаний, апелляции/взлом/платежи у человека).
2. Косвенные метки из ответов админов в valid (поле violation, verdict) — грубая оценка на всех тикетах.
3. Эталон data/eval_gold.jsonl — 80 тикетов, размеченных вручную по taxonomy.json.
"""
import argparse
import collections
import json
from pathlib import Path

from app.services.escalation import forbidden_in_reply, misaddressed_reply, sensitive_reasons
from app.services.extractor import extract_rule_refs
from app.services.rules import rule_exists
from scripts.dataset import read_jsonl

# Категория по пункту, за который админ наказал (поле violation)
VIOLATION_TO_CATEGORY = {
    "2.1": "complaint.insult", "2.4": "complaint.insult", "2.6": "complaint.insult",
    "5.2": "complaint.cheat",
    "5.3": "complaint.dm", "5.4": "complaint.dm",
    "4.1": "complaint.scam", "4.2": "complaint.scam",
    "2.3": "complaint.rmt_ads", "4.4": "complaint.rmt_ads",
}


def proxy_category(violation: str | None) -> str | None:
    if not violation:
        return None
    for ref in extract_rule_refs(violation):
        if ref.startswith("GZ "):
            return "complaint.dm"
        if ref in VIOLATION_TO_CATEGORY:
            return VIOLATION_TO_CATEGORY[ref]
    if violation.strip().upper() in ("DM", "ДМ"):
        return "complaint.dm"
    return None


def pct(part: int, total: int) -> str:
    return f"{part}/{total} = {part / total:.1%}" if total else "нет данных"


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = {gold for gold, _ in pairs}
    scores = []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="outputs/predictions_valid.jsonl")
    parser.add_argument("--data", default="data/tickets_valid.jsonl")
    parser.add_argument("--gold", default="data/eval_gold.jsonl")
    args = parser.parse_args()

    predictions = {p["ticket_id"]: p for p in read_jsonl(args.pred)}
    tickets = {t["ticket_id"]: t for t in read_jsonl(args.data)}
    print(f"Предсказаний: {len(predictions)} из {len(tickets)} тикетов\n")

    def player_text(ticket_id: str) -> str:
        return "\n".join(m.get("text") or "" for m in tickets[ticket_id]["messages"] if m["role"] == "player")

    # ---------- 1. Инварианты безопасности ----------
    print("=== 1. Безопасность (цель — 100%)")
    preds = list(predictions.values())
    cited = [c for p in preds for c in p["draft_reply"]["cited"] + p["extracted"]["rule_refs"]]
    print(f"Цитаты существуют в rules.json:      {pct(sum(map(rule_exists, cited)), len(cited))}")
    print(f"Без запрещённых обещаний в черновике: "
          f"{pct(sum(1 for p in preds if not forbidden_in_reply(p['draft_reply']['text'])), len(preds))}")
    print(f"Черновик адресован заявителю:        "
          f"{pct(sum(1 for p in preds if not misaddressed_reply(p['category'], p['draft_reply']['text'])), len(preds))}")
    print(f"Язык черновика = язык обращения:     "
          f"{pct(sum(1 for p in preds if p['draft_reply']['lang'] == p['lang']), len(preds))}")
    print(f"Категория соответствует виду тикета: "
          f"{pct(sum(1 for t, p in predictions.items() if p['category'].startswith(tickets[t]['kind'])), len(preds))}")
    sensitive = [t for t in predictions if t in tickets and sensitive_reasons(player_text(t))]
    print(f"Апелляции/взлом/платежи → человеку:  "
          f"{pct(sum(1 for t in sensitive if predictions[t]['needs_human']), len(sensitive))}")

    # ---------- 2. Косвенные метки из ответов админов ----------
    print("\n=== 2. Косвенные метки из ответов админов (грубая оценка)")
    pairs = [(proxy_category(tickets[t].get("violation")), p["category"]) for t, p in predictions.items()]
    pairs = [(g, p) for g, p in pairs if g]
    print(f"Категория совпала с пунктом наказания:  {pct(sum(1 for g, p in pairs if g == p), len(pairs))}")
    appeals = [t for t in predictions if tickets[t].get("verdict") in ("appeal_approved", "appeal_rejected")]
    found = sum(1 for t in appeals if predictions[t]["category"] == "complaint.admin"
                or any("Апелляция" in r for r in predictions[t]["reasons"]))
    print(f"Апелляции распознаны:                    {pct(found, len(appeals))}")
    dups = [t for t in predictions if tickets[t].get("verdict") == "duplicate"]
    print(f"Дубликаты (verdict=duplicate) отмечены:  {pct(sum(predictions[t]['is_spam_or_duplicate'] for t in dups), len(dups))}"
          "  (ожидаемо низко: отмечаем только дословные повторы)")

    # ---------- 2б. По языкам (язык — как определила система) ----------
    print("\n=== 2б. По языкам")
    by_lang = collections.defaultdict(list)
    for t, p in predictions.items():
        by_lang[p["lang"]].append(t)
    for lang, ids in sorted(by_lang.items(), key=lambda x: -len(x[1])):
        lang_pairs = [(proxy_category(tickets[t].get("violation")), predictions[t]["category"]) for t in ids]
        lang_pairs = [(g, p) for g, p in lang_pairs if g]
        lang_appeals = [t for t in ids if tickets[t].get("verdict") in ("appeal_approved", "appeal_rejected")]
        found_appeals = sum(1 for t in lang_appeals if predictions[t]["category"] == "complaint.admin"
                            or any("Апелляция" in r for r in predictions[t]["reasons"]))
        print(f"  {lang}: тикетов {len(ids):4} | категория = пункт наказания "
              f"{pct(sum(1 for g, p in lang_pairs if g == p), len(lang_pairs))} | "
              f"апелляции {pct(found_appeals, len(lang_appeals))} | "
              f"человеку {pct(sum(predictions[t]['needs_human'] for t in ids), len(ids))}")

    # ---------- 3. Эталон ----------
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"\nЭталон {gold_path} не найден — пропускаем")
        return
    gold = [g for g in read_jsonl(gold_path) if g["ticket_id"] in predictions]
    print(f"\n=== 3. Эталон ({len(gold)} размеченных тикетов есть в предсказаниях)")
    if not gold:
        return
    cat_pairs = [(g["category"], predictions[g["ticket_id"]]["category"]) for g in gold]
    print(f"Категория, accuracy:   {pct(sum(1 for a, b in cat_pairs if a == b), len(gold))}")
    print(f"Категория, macro-F1:   {macro_f1(cat_pairs):.2f}")
    print(f"Приоритет, accuracy:   {pct(sum(1 for g in gold if g['priority'] == predictions[g['ticket_id']]['priority']), len(gold))}")
    pri_pairs = [(g["priority"], predictions[g["ticket_id"]]["priority"]) for g in gold]
    print(f"Приоритет, macro-F1:   {macro_f1(pri_pairs):.2f}  (по классам эталона: "
          f"{', '.join(sorted({g for g, _ in pri_pairs}))})")
    pri_errors = collections.Counter((a, b) for a, b in pri_pairs if a != b)
    if pri_errors:
        print("  ошибки приоритета (эталон → предсказание): "
              + ", ".join(f"{a}→{b}: {n}" for (a, b), n in pri_errors.most_common()))
    print(f"needs_human, accuracy: {pct(sum(1 for g in gold if g['needs_human'] == predictions[g['ticket_id']]['needs_human']), len(gold))}")
    must_human = [g for g in gold if g["needs_human"]]
    print(f"needs_human, recall:   {pct(sum(1 for g in must_human if predictions[g['ticket_id']]['needs_human']), len(must_human))}"
          "  (пропуск человека опаснее лишней эскалации)")
    print(f"Язык, accuracy:        {pct(sum(1 for g in gold if g['lang'] == predictions[g['ticket_id']]['lang']), len(gold))}")

    errors = collections.Counter((a, b) for a, b in cat_pairs if a != b)
    if errors:
        print("Частые ошибки категории (эталон → предсказание):")
        for (a, b), n in errors.most_common(6):
            print(f"  {a} → {b}: {n}")


if __name__ == "__main__":
    main()
