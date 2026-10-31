"""Вся логика обработки одного обращения.

Вынесена из main.py, чтобы одну и ту же функцию вызывали и API (/triage),
и scripts/predict.py для JSONL — без HTTP и без дублирования кода.

Порядок:
1. Код без LLM: язык, факты, спам, дубликаты.
2. Пустое обращение — сразу шаблонный ответ, LLM не вызываем (бесплатно).
3. Один вызов LLM с правилами-кандидатами (RAG).
4. Проверка ответа LLM кодом: цитаты, категория, эскалация.
5. Если LLM недоступна или ответила мусором — безопасный ответ, needs_human=true.
"""
from app.schemas import DraftReply, Extracted, TriageRequest, TriageResponse
from app.services import llm
from app.services.classifier import KeywordResult, category_priority, classify_by_keywords
from app.services.deduplication import DuplicateDetector, is_spam
from app.services.escalation import (check_priority, decide_needs_human, disagreement_reason,
                                     forbidden_in_reply, has_p0_signal, misaddressed_reply, sensitive_reasons)
from app.services.extractor import extract_nickname, extract_rule_refs, extract_server, has_evidence
from app.services.generator import template_for, template_reply
from app.services.language import detect_language
from app.services.normalize import normalize_text
from app.services.rag import candidate_rules, load_taxonomy, relevant_citations

duplicates = DuplicateDetector()


def triage(request: TriageRequest) -> TriageResponse:
    player_messages = [m for m in request.messages if m.role == "player"]
    # Нормализация Unicode: «Ｏｓｋ», «𝐨𝐬𝐤», невидимые символы внутри слов -> обычный текст
    player_text = normalize_text("\n".join(m.text for m in player_messages))
    accused_comment = normalize_text("\n".join(m.text for m in request.messages if m.role == "reported_player"))
    attachments = sum(m.attachments for m in player_messages)

    lang, lang_confident = detect_language(player_text, request.server_lang)
    extracted = Extracted(
        nickname=request.reported_nick or extract_nickname(player_text),
        rule_refs=extract_rule_refs(player_text),
        has_evidence=has_evidence(player_text, attachments),
        server=extract_server(player_text),
    )
    reasons: list[str] = []
    if not lang_confident:
        reasons.append("Язык не определён по тексту, выбран язык по умолчанию")

    spam = is_spam(player_text, attachments)
    duplicate = duplicates.check_and_remember(
        request.reporter_id, extracted.nickname, player_text, request.created_at
    )

    # Пустое обращение: просим описать подробнее, человек не нужен
    if spam:
        return TriageResponse(
            lang=lang,
            category="complaint.invalid" if request.kind == "complaint" else "report.offtopic",
            priority="p3",
            is_spam_or_duplicate=True,
            extracted=extracted,
            draft_reply=DraftReply(lang=lang, text=template_reply("need_details", lang), cited=[]),
            needs_human=False,
            confidence=0.9,
            reasons=reasons + ["Сообщение без содержания и без доказательств"],
        )

    if duplicate:
        reasons.append("Дословный повтор обращения того же игрока за 24 часа")

    # Короткое однозначное обращение («оск», «опру», «помогите») — без LLM, экономим квоту
    keywords = classify_by_keywords(request.kind, player_text, extracted.rule_refs)
    if keywords and keywords.confident:
        return _keyword_response(keywords, lang, extracted, duplicate, player_text, 0.7,
                                 reasons + ["Классифицировано по ключевым словам, LLM не вызывалась"])

    # RAG: все 266 пунктов ранжируются по тексту и подсказке ключевых слов, в LLM уходит top-k
    rules = candidate_rules(request.kind, extracted.rule_refs, text=player_text,
                            category_hint=keywords.category if keywords else None)
    call = llm.classify(
        kind=request.kind,
        text=player_text,
        lang=lang,
        attachments=attachments,
        player_refs=extracted.rule_refs,
        rules=rules,
        accused_comment=accused_comment,
    )
    if isinstance(call, llm.LLMFailure):
        if keywords:  # LLM недоступна, но слова подсказывают категорию
            return _keyword_response(keywords, lang, extracted, duplicate, player_text, 0.5,
                                     reasons + [call.reason, "Категория определена по ключевым словам"])
        return _fallback(request.kind, lang, extracted, duplicate,
                         reasons + sensitive_reasons(player_text) + [call.reason])

    # Ollama — экспериментальная опция (решение команды 2026-10-31 по замеру Disabled vs Ollama): её ответ не меняет
    # категорию, приоритет, needs_human и черновик — только записывается в reasons для сравнения
    settings = llm.current_settings()
    if settings and settings.provider == "ollama":
        suggestion = (f"Ollama (эксперимент, на ответ не влияет): {call.result.category}, {call.result.priority}, "
                      f"needs_human={call.result.needs_human}")
        if keywords:
            return _keyword_response(keywords, lang, extracted, duplicate, player_text, 0.5,
                                     reasons + ["Категория определена по ключевым словам", suggestion])
        return _fallback(request.kind, lang, extracted, duplicate, reasons + sensitive_reasons(player_text) + [suggestion])

    # Как используется каждое поле ответа LLM (модель предлагает — код решает):
    #   category    -> берётся, если совпадает с видом обращения, иначе безопасный ответ
    #   priority    -> берётся, но p0 без признаков взлома/платежа/угроз понижается до p1 (check_priority)
    #   cited       -> остаются только ID из правил, которые мы сами дали модели (RAG-кандидаты)
    #   needs_human -> может только ДОБАВИТЬ эскалацию; правила в коде её не снимают (decide_needs_human)
    #   confidence  -> приводится к [0, 1]; ниже 0.5 — к администратору
    #   reply       -> проверяется на запрещённые обещания; при нарушении — нейтральный шаблон
    #   reason      -> пояснение для администратора, помечается как мнение LLM
    #   язык ответа модель не выбирает: lang определяет код (detect_language)
    result = call.result
    # Программная проверка цитат: только пункты, которые мы сами дали модели
    allowed = {rule.id for rule in rules}
    cited = [rule_id for rule_id in dict.fromkeys(result.cited) if rule_id in allowed]
    if len(cited) < len(set(result.cited)):
        reasons.append("Часть цитат LLM отброшена: таких пунктов нет среди проверенных правил")
    # Релевантность: пункт должен относиться к категории, быть назван игроком или совпадать по словам
    relevant = relevant_citations(cited, result.category, extracted.rule_refs, player_text)
    if len(relevant) < len(cited):
        reasons.append("Цитаты не по теме обращения удалены: " + ", ".join(c for c in cited if c not in relevant))
    cited = relevant

    # Категория должна соответствовать виду обращения
    if not result.category.startswith(request.kind + "."):
        return _fallback(request.kind, lang, extracted, duplicate, reasons + ["LLM вернула категорию не того вида"])

    confidence = min(max(result.confidence, 0.0), 1.0)
    priority, priority_reason = check_priority(result.category, result.priority, player_text)
    if priority_reason:
        reasons.append(priority_reason)
    # Приоритет — по правилам категории. p0 правило категории даёт только при признаках взлома/платежа/угроз:
    # qwen2.5:3b назвала «смену ника» взломом аккаунта (report.account) — это не повод для p0
    rule_priority = category_priority(result.category, player_text)
    if rule_priority == "p0" and not has_p0_signal(player_text):
        rule_priority = priority
    if priority != "p0" and priority != rule_priority:
        reasons.append(f"Приоритет {priority} от LLM заменён на {rule_priority} по правилам категории")
        priority = rule_priority
    needs_human, escalation_reasons = decide_needs_human(
        result.category, priority, result.needs_human, confidence,
        player_text, load_taxonomy()["needs_human_always"],
    )
    reasons += escalation_reasons
    # LLM не может «уйти» от эскалации, выбрав категорию без участия админа вопреки ключевым словам
    disagreement = disagreement_reason(result.category, keywords.category if keywords else None)
    if disagreement:
        needs_human = True
        reasons.append(disagreement)

    reply = result.reply
    hint = keywords.category if keywords else None  # чтобы не ответить пострадавшему как апеллянту
    # Черновик с запрещённым обещанием не отдаём: заменяем нейтральным шаблоном
    violations = forbidden_in_reply(reply)
    wrong_addressee = misaddressed_reply(result.category, reply)
    if wrong_addressee:
        violations.append(wrong_addressee)
    if violations:
        reply, cited, needs_human = template_reply(template_for(result.category, player_text, extracted.has_evidence, hint), lang), [], True
        reasons.append("Черновик LLM заменён шаблоном: " + "; ".join(violations))
    elif disagreement:
        # Черновик построен на категории, с которой не согласны ключевые слова, — не доверяем ему
        reply, cited = template_reply(template_for(keywords.category, player_text, extracted.has_evidence), lang), []
        reasons.append("Черновик LLM заменён шаблоном: категория спорная")
    else:
        # Язык: черновику модели доверяем не для всех языков (LLM_DRAFT_LANGS), и он должен быть на языке игрока
        language_problem = _draft_language_problem(reply, lang)
        if language_problem:
            # Цитаты уже проверены (существуют и по теме) — оставляем их администратору, меняется только текст
            reply = template_reply(template_for(result.category, player_text, extracted.has_evidence, hint), lang)
            reasons.append(language_problem)
    if duplicate:
        reply, cited = template_reply("duplicate", lang), []

    return TriageResponse(
        lang=lang,
        category=result.category,
        priority=priority,
        is_spam_or_duplicate=duplicate,
        extracted=extracted,
        draft_reply=DraftReply(lang=lang, text=reply, cited=cited),
        needs_human=needs_human,
        confidence=confidence,
        reasons=reasons + [f"Пояснение LLM: {result.reason}"],
    )


def _draft_language_problem(reply: str, lang: str) -> str | None:
    """Причина заменить черновик модели шаблоном на языке игрока, или None.
    Шаблоны написаны вручную на 6 языках, поэтому это безопасный fallback для слабых языков модели."""
    settings = llm.current_settings()
    if settings and lang not in settings.draft_langs:
        return f"Черновик модели для языка {lang} отключён (LLM_DRAFT_LANGS) — использован шаблон"
    reply_lang, confident = detect_language(reply)
    if confident and reply_lang != lang:
        return f"Модель ответила на языке {reply_lang} вместо {lang} — использован шаблон"
    return None


def _keyword_response(
    keywords: KeywordResult,
    lang: str,
    extracted: Extracted,
    duplicate: bool,
    player_text: str,
    confidence: float,
    reasons: list[str],
) -> TriageResponse:
    """Ответ без LLM: категория по словам, черновик — шаблон, цитаты — ссылки игрока или лучший по RAG пункт категории."""
    priority, priority_reason = check_priority(keywords.category, keywords.priority, player_text)
    needs_human, escalation_reasons = decide_needs_human(
        keywords.category, priority, False, confidence, player_text, load_taxonomy()["needs_human_always"]
    )
    if duplicate:
        template = "duplicate"
    elif keywords.category in ("report.offtopic", "complaint.invalid"):
        template = "need_details"
    else:
        template = template_for(keywords.category, player_text, extracted.has_evidence)  # адресный шаблон: жалоба / апелляция / обращение
    return TriageResponse(
        lang=lang,
        category=keywords.category,
        priority=priority,
        is_spam_or_duplicate=duplicate,
        extracted=extracted,
        draft_reply=DraftReply(lang=lang, text=template_reply(template, lang),
                               cited=[] if duplicate else _rag_citations(keywords.category, extracted.rule_refs, player_text)),
        needs_human=needs_human,
        confidence=confidence,
        reasons=reasons + ([priority_reason] if priority_reason else []) + escalation_reasons,
    )


def _rag_citations(category: str, player_refs: list[str], player_text: str) -> list[str]:
    """Цитаты основного режима (без LLM). Ссылки игрока (уже проверены по rules.json) — как есть. Если их нет,
    для жалоб на нарушение берётся лучший по RAG пункт, относящийся к категории (relevant_citations: taxonomy.json,
    зелёная зона — только при её упоминании). Для report.* не цитируем: таксономия связывает «застрял» с п. 3.1
    «репорт не по назначению» — игрок прочитал бы это как обвинение."""
    if player_refs:
        return player_refs
    if not category.startswith("complaint.") or category in ("complaint.other", "complaint.invalid"):
        return []
    kind = category.split(".")[0]
    ranked = [rule.id for rule in candidate_rules(kind, [], text=player_text, category_hint=category)]
    return relevant_citations(ranked, category, [], player_text)[:1]


def _fallback(kind: str, lang: str, extracted: Extracted, duplicate: bool, reasons: list[str]) -> TriageResponse:
    """Безопасный ответ, когда LLM недоступна: ничего не решаем, отдаём человеку."""
    return TriageResponse(
        lang=lang,
        category="complaint.other" if kind == "complaint" else "report.offtopic",
        priority="p2",
        is_spam_or_duplicate=duplicate,
        extracted=extracted,
        draft_reply=DraftReply(lang=lang, text=template_reply("duplicate" if duplicate else "received", lang), cited=[]),
        needs_human=True,
        confidence=0.0,
        reasons=reasons + ["Автоматическая классификация недоступна, нужна проверка администратором"],
    )
