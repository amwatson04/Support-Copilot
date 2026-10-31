"""RAG: ранжирование пунктов правил. Метрики Recall@k на данных — scripts/eval_rag.py."""
from app.services.rag import TOP_K, candidate_rules, rank_rules
from app.services.rules import load_rules


def test_player_refs_come_first_and_are_not_cut():
    refs = ["ЖБ 1.1.2", "ПГО 1.4", "FW 1.7.1", "2.1", "5.2", "GZ 1.2", "4.1", "2.3", "ОПГ 2.1"]  # 9 > TOP_K
    ids = [r.id for r in candidate_rules("complaint", refs, text="оск")]
    assert ids[:len(refs)] == refs


def test_only_existing_rules_are_returned():
    registry = load_rules()
    ids = rank_rules("complaint", "любой текст", ["9.9", "2.1"])
    assert "9.9" not in ids and ids[0] == "2.1"
    assert set(ids) == set(registry)  # ранжируются все 266 пунктов, ничего не выдумывается


def test_slang_finds_the_right_rule():
    def top(text, hint=None):
        return [r.id for r in candidate_rules("complaint", [], text=text, category_hint=hint)]
    assert "2.1" in top("оск родни", "complaint.insult")[:3]
    assert "5.2" in top("у него софт, аим", "complaint.cheat")[:3]
    assert "4.1" in top("кинул на машину, скам", "complaint.scam")[:3]


def test_rule_outside_taxonomy_can_be_retrieved():
    # Пункт про никнеймы (1.6) не входит в списки таксономии — раньше он не мог попасть к модели
    ids = [r.id for r in candidate_rules("complaint", [], text="игрок взял никнейм, схожий с ником администратора")]
    assert "1.6" in ids and len(ids) == TOP_K


def test_backward_compatible_call():
    ids = [r.id for r in candidate_rules("report", [])]
    assert ids[:2] == ["3.1", "1.4"] and len(ids) == TOP_K


# ---------- Релевантность цитат (слепая оценка черновиков 2026-10-31) ----------

from app.services.rag import relevant_citations  # noqa: E402


def test_off_topic_citations_removed():
    text = "Договорились обменять машину на деньги, он забрал машину и вышел из игры"
    assert relevant_citations(["5.4", "2.4", "4.1"], "complaint.scam", [], text) == ["4.1"]


def test_green_zone_cited_only_if_mentioned():
    assert relevant_citations(["GZ 1.1"], "complaint.dm", [], "Меня убили у банка без причины") == []
    assert relevant_citations(["GZ 1.1"], "complaint.dm", [], "Меня убили в зелёной зоне") == ["GZ 1.1"]
    assert relevant_citations(["GZ 1.1"], "complaint.dm", ["GZ 1.1"], "1.1 gz") == ["GZ 1.1"]


def test_player_ref_always_kept_and_appeal_trimmed():
    text = "Мне выдали мут ни за что, прошу пересмотреть"
    kept = relevant_citations(["2.1", "2.4", "5.2", "5.3", "3.4"], "complaint.admin", [], text)
    assert kept == ["3.4"]  # 3.4 — пункт категории «жалоба на администратора»
    assert relevant_citations(["2.6"], "complaint.insult", ["2.6"], "2.6") == ["2.6"]


def test_word_overlap_alone_is_not_a_citation():
    # FW 1.17 в апелляции проходил только по совпадению слов — теперь нужна категория или ссылка игрока
    from app.services.rag import relevant_citations
    kept = relevant_citations(["FW 1.17", "3.4"], "complaint.admin", [], "не согласен с наказанием, я только защищался от другого игрока")
    assert kept == ["3.4"]


def test_main_mode_cites_rag_rule_of_category():
    # Без LLM: игрок не назвал пункт → цитируется лучший по RAG пункт категории (а не пусто)
    from app import pipeline
    from app.schemas import TriageRequest
    out = pipeline.triage(TriageRequest(kind="complaint", messages=[
        {"text": "Он весь вечер оскорблял мою маму и родственников в общем чате", "attachments": 1}]))
    assert out.category == "complaint.insult"
    assert len(out.draft_reply.cited) == 1 and out.draft_reply.cited[0] in {"2.1", "2.4", "2.6"}
    help_out = pipeline.triage(TriageRequest(kind="report", messages=[{"text": "застрял в текстурах помогите"}]))
    assert help_out.draft_reply.cited == []  # report.*: таксономия даёт 3.1 «репорт не по назначению» — не цитируем
