"""Подбор пунктов правил, которые передаём LLM (Retrieval в RAG).

Почему без векторной базы:
- правил всего 266 коротких пунктов (~60 КБ) — BM25 в памяти считается за миллисекунды;
- правила на русском, а игроки пишут коротко, сленгом и на нескольких языках —
  эмбеддинги потребовали бы ещё одну модель, а выигрыш не проверен.

Как ранжируем (все 266 пунктов, а не фиксированный список):
1. Пункты, которые игрок назвал сам («2.1 оп»), — всегда первыми.
2. Счёт = BM25 по словам обращения + бонус пунктам категории, которую подсказали ключевые слова
   (taxonomy.json), + небольшой бонус пунктам любой категории этого вида обращения.
3. Сленг разворачивается в слова правил: «оск» → «оскорбление», «дм» → «убийство урон», «софт» → «стороннее ПО».
LLM может цитировать только то, что попало в top-k.
Качество измеряется scripts/eval_rag.py по пункту, за который реально наказал админ (Recall@k).
"""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.services.rules import ALIASES, RulePoint, load_rules, make_id

TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "taxonomy.json"

# Параметры подобраны на tickets_train (12 048 жалоб с пунктом наказания), проверены на valid: scripts/eval_rag.py
TOP_K = 8               # сколько пунктов отдаём LLM
BM25_WEIGHT = 0.3       # вклад совпадения слов
CATEGORY_BONUS = 2.0    # пункт из правил категории, подсказанной ключевыми словами
KIND_PRIOR = 8.0        # пункт из таксономии этого вида обращения; убывает по порядку таксономии
KIND_DECAY = 0.5        # насколько последний пункт таксономии слабее первого
LEXICAL_SLOTS = 2       # мест в top-k, зарезервированных под лучшие совпадения BM25 (правила вне таксономии)
MIN_LEXICAL = 2.0       # минимальный счёт BM25 для зарезервированного места
_STEM = 5               # грубая основа слова: первые 5 букв (русская морфология)
_BM25_K1, _BM25_B = 1.2, 0.75

_WORD = re.compile(r"[a-zа-яёәғқңөұүһі]+", re.IGNORECASE)

# Сленг игроков -> слова, которыми написаны правила. Подобрано по частым словам в tickets_train.
SLANG = {
    "оск": "оскорбление", "оскор": "оскорбление", "оскар": "оскорбление", "osk": "оскорбление",
    "обзыв": "оскорбление", "мат": "ненормативная лексика", "матер": "ненормативная лексика",
    "родн": "родственников", "род": "родственников", "мам": "родственников", "rodn": "родственников",
    "дм": "убийство урон", "dm": "убийство урон", "дб": "убийство урон транспорт", "db": "убийство урон транспорт",
    "тк": "убийство урон", "tk": "убийство урон", "ск": "убийство урон", "убил": "убийство", "убив": "убийство",
    "зз": "зелёной зоне", "gz": "зелёной зоне", "гз": "зелёной зоне", "связк": "верёвок похищения",
    "чит": "стороннего программного обеспечения", "софт": "стороннего программного обеспечения",
    "soft": "стороннего программного обеспечения", "аим": "стороннего программного обеспечения",
    "скам": "мошенничество обман", "scam": "мошенничество обман", "кинул": "мошенничество обман",
    "обман": "мошенничество обман", "реклам": "реклама", "продае": "сделки реальные деньги",
    "ник": "никнеймы", "нрп": "rp ситуация", "нонрп": "rp ситуация", "nonrp": "rp ситуация", "nrp": "rp ситуация",
    "админ": "администрации", "угроз": "угрожать", "расизм": "расизм", "нацизм": "пропаганда",
}


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _expand(taxonomy_ref: str) -> list[str]:
    """«2.1» -> ["2.1"], «ЖБ 1.1.2» -> ["ЖБ 1.1.2"], «GZ» -> все пункты зелёной зоны."""
    rules = load_rules()
    parts = taxonomy_ref.split()
    if len(parts) == 1 and parts[0].upper() in ALIASES and parts[0][0].isalpha():
        prefix = ALIASES[parts[0].upper()]
        return [rule_id for rule_id in rules if rule_id.startswith(prefix + " ")]
    if len(parts) == 2 and parts[0].upper() in ALIASES:
        return [make_id(ALIASES[parts[0].upper()], parts[1])]
    return [taxonomy_ref]


def categories_for(kind: str) -> list[dict]:
    return [c for c in load_taxonomy()["categories"] if c["kind"] == kind]


@lru_cache(maxsize=2)
def _kind_rule_ids(kind: str) -> tuple[str, ...]:
    ids: list[str] = []
    for category in categories_for(kind):
        for ref in category["rules"]:
            ids.extend(_expand(ref))
    return tuple(dict.fromkeys(ids))


@lru_cache(maxsize=32)
def _category_rule_ids(category: str) -> frozenset[str]:
    for item in load_taxonomy()["categories"]:
        if item["code"] == category:
            return frozenset(i for ref in item["rules"] for i in _expand(ref))
    return frozenset()


# ---------- BM25 ----------

def _terms(text: str) -> list[str]:
    """Слова → грубые основы; сленг разворачивается в слова правил."""
    words: list[str] = []
    for word in _WORD.findall(text.lower()):
        for slang, expansion in SLANG.items():
            if word.startswith(slang) and len(word) <= len(slang) + 4:
                words.extend(expansion.split())
                break
        words.append(word)
    return [w[:_STEM] for w in words if len(w) >= 3]


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, Counter], dict[str, int], dict[str, float], float]:
    docs = {rid: Counter(_terms(point.text)) for rid, point in load_rules().items()}
    lengths = {rid: sum(c.values()) for rid, c in docs.items()}
    df = Counter(term for c in docs.values() for term in c)
    n = len(docs)
    idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
    return docs, lengths, idf, sum(lengths.values()) / n


def _bm25(query: list[str]) -> dict[str, float]:
    docs, lengths, idf, avg_len = _index()
    scores: dict[str, float] = {}
    for rid, counts in docs.items():
        score = 0.0
        for term in set(query):
            tf = counts.get(term)
            if tf:
                score += idf[term] * tf * (_BM25_K1 + 1) / (
                    tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * lengths[rid] / avg_len))
        if score:
            scores[rid] = score
    return scores


def rank_rules(kind: str, text: str = "", player_refs: list[str] | None = None,
               category_hint: str | None = None, k: int = TOP_K) -> list[str]:
    """Все пункты правил по убыванию релевантности (ID).

    Порядок: ссылки игрока; затем (k - LEXICAL_SLOTS) мест по общему счёту (приоритет таксономии + BM25);
    затем LEXICAL_SLOTS мест под лучшие совпадения слов, даже если пункта нет в таксономии; затем остальные."""
    rules = load_rules()
    refs = [r for r in (player_refs or []) if r in rules]
    scores = _bm25(_terms(text)) if text else {}
    hinted = _category_rule_ids(category_hint) if category_hint else frozenset()
    taxonomy = _kind_rule_ids(kind)
    position = {rid: i for i, rid in enumerate(taxonomy)}

    def total(rid: str) -> float:
        prior = 0.0
        if rid in position:
            prior = KIND_PRIOR * (1 - KIND_DECAY * position[rid] / len(taxonomy))
        return BM25_WEIGHT * scores.get(rid, 0.0) + CATEGORY_BONUS * (rid in hinted) + prior

    ranked = sorted((rid for rid in rules if rid not in refs),
                    key=lambda rid: (-total(rid), position.get(rid, len(position))))
    head = ranked[:max(k - LEXICAL_SLOTS - len(refs), 0)]
    lexical = [rid for rid, s in sorted(scores.items(), key=lambda kv: -kv[1])
               if s >= MIN_LEXICAL and rid not in refs and rid not in head][:LEXICAL_SLOTS]
    rest = [rid for rid in ranked if rid not in head and rid not in lexical]
    return refs + head + lexical + rest


_GREEN_ZONE = re.compile(r"зел[её]н|\bзз\b|\bgz\b|\bгз\b|green|yashil", re.IGNORECASE)


def relevant_citations(cited: list[str], category: str, player_refs: list[str], text: str) -> list[str]:
    """Оставляет только цитаты, которые связаны с обращением (слепая оценка 2026-10-31: модель цитировала
    5.4 и 2.4 в жалобе на мошенничество, 6 пунктов в апелляции, «зелёную зону» без её упоминания).

    Цитата остаётся, если пункт: назван игроком; ИЛИ относится к итоговой категории по taxonomy.json.
    Пункты зелёной зоны — только если она упомянута. Совпадения слов (BM25) для цитаты недостаточно:
    повторный замер Ollama 2026-10-31 — так проходили FW 1.17 в апелляции и ВБ 3.4 в жалобе на обман."""
    refs = set(player_refs)
    allowed = refs | _category_rule_ids(category)
    green = bool(_GREEN_ZONE.search(text))
    return [rid for rid in cited
            if rid in allowed and (rid in refs or not rid.startswith("GZ ") or green)]


def candidate_rules(kind: str, player_refs: list[str], text: str = "",
                    category_hint: str | None = None, k: int = TOP_K) -> list[RulePoint]:
    """Пункты правил, которые LLM может процитировать в этом обращении (top-k после ранжирования)."""
    rules = load_rules()
    ids = rank_rules(kind, text, player_refs, category_hint, k)
    # ссылки игрока не обрезаются лимитом k
    limit = max(k, len([r for r in player_refs if r in rules]))
    return [rules[i] for i in ids[:limit]]
