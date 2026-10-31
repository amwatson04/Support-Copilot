"""Оценка retrieval: находит ли RAG пункт, за который реально наказал администратор.

Эталон для каждого тикета — пункты из поля `violation` (вердикт админа), только одобренные жалобы,
где пункт распознан. Это не разметка релевантности всех правил, а «хотя бы этот пункт должен найтись».

Запуск:
    python -m scripts.eval_rag --file data/tickets_train.jsonl.gz     # подбор параметров (dev)
    python -m scripts.eval_rag --file data/tickets_valid.jsonl        # итоговая проверка (test)

Сравнивает старый фиксированный набор кандидатов и новое ранжирование.
"""
import argparse
import collections

from app.services.classifier import classify_by_keywords
from app.services.extractor import extract_rule_refs
from app.services.rag import _kind_rule_ids, rank_rules
from app.services.rules import rule_exists
from scripts.dataset import read_jsonl

KS = (1, 3, 5, 8)


def baseline(kind: str, refs: list[str]) -> list[str]:
    """Как было до 2026-10-31: ссылки игрока + фиксированный список пунктов категорий вида."""
    return list(dict.fromkeys([*[r for r in refs if rule_exists(r)], *_kind_rule_ids(kind)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/tickets_valid.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stats = {name: collections.Counter() for name in ("baseline", "ranked")}
    per_doc = collections.defaultdict(collections.Counter)
    n = n_no_refs = 0
    for record in read_jsonl(args.file):
        if record["kind"] != "complaint":
            continue
        gold = set(extract_rule_refs(record.get("violation") or ""))
        if not gold:
            continue
        text = "\n".join(m.get("text") or "" for m in record["messages"] if m["role"] == "player")
        refs = extract_rule_refs(text)
        hint = classify_by_keywords("complaint", text, refs)
        lists = {
            "baseline": baseline("complaint", refs),
            "ranked": rank_rules("complaint", text, refs, hint.category if hint else None),
        }
        n += 1
        no_refs = not (gold & set(refs))
        n_no_refs += no_refs
        for name, ids in lists.items():
            for k in KS:
                hit = bool(gold & set(ids[:k]))
                stats[name][f"R@{k}"] += hit
                stats[name][f"P@{k}"] += len(gold & set(ids[:k])) / k
                if no_refs:
                    stats[name][f"R@{k}|без ссылки игрока"] += hit
            stats[name]["в списке вообще"] += bool(gold & set(ids))
            rank = next((i for i, rid in enumerate(ids) if rid in gold), None)
            stats[name]["MRR"] += 1 / (rank + 1) if rank is not None else 0
        doc = next(iter(gold)).split()[0] if " " in next(iter(gold)) else "ОП"
        per_doc[doc]["n"] += 1
        per_doc[doc]["baseline R@8"] += bool(gold & set(lists["baseline"][:8]))
        per_doc[doc]["ranked R@8"] += bool(gold & set(lists["ranked"][:8]))
        if args.limit and n >= args.limit:
            break

    print(f"Файл: {args.file}. Жалоб с распознанным пунктом наказания: {n} (из них без ссылки игрока: {n_no_refs})")
    print(f"{'метрика':28} {'было (фикс. список)':>20} {'стало (ранжирование)':>21}")
    keys = [f"R@{k}" for k in KS] + [f"R@{k}|без ссылки игрока" for k in KS] + [f"P@{k}" for k in KS] + \
           ["MRR", "в списке вообще"]
    for key in keys:
        denom = n_no_refs if "без ссылки" in key else n
        row = [stats[name][key] / denom if denom else 0 for name in ("baseline", "ranked")]
        print(f"{key:28} {row[0]:>20.3f} {row[1]:>21.3f}")
    print("\nR@8 по документам (пункт наказания):")
    for doc, c in sorted(per_doc.items(), key=lambda x: -x[1]["n"]):
        print(f"  {doc:6} n={c['n']:5}  было {c['baseline R@8'] / c['n']:.2f}  стало {c['ranked R@8'] / c['n']:.2f}")


if __name__ == "__main__":
    main()
