"""Шаблонные черновики ответов на 6 языках.

Используются, когда черновик LLM не нужен или ему нельзя доверять:
- пустое обращение или спам («need_details»);
- классификация по ключевым словам без LLM, LLM недоступна («received»);
- дословный повтор обращения («duplicate»);
- черновик LLM содержал запрещённое обещание или был адресован не тому — заменяется шаблоном
  по категории (template_for): «complaint_received» — жалоба на другого игрока, «appeal_received» — апелляция.
Черновик LLM формирует app/services/llm.py, проверяет — app/pipeline.py.
"""
from app.services.escalation import SENSITIVE_PATTERNS

TEMPLATES = {
    # Жалоба на другого игрока: обращаемся к заявителю, ничего не утверждаем о нарушении
    "complaint_received": {
        "ru": "Здравствуйте! Спасибо за жалобу. Администратор проверит приложенные доказательства и примет решение.",
        "kk": "Сәлеметсіз бе! Шағымыңыз үшін рақмет. Әкімші тіркелген дәлелдерді тексеріп, шешім қабылдайды.",
        "uz": "Assalomu alaykum! Shikoyatingiz uchun rahmat. Administrator ilova qilingan dalillarni tekshirib, "
              "qaror qabul qiladi.",
        "en": "Hello! Thank you for your report. An administrator will review the evidence you provided "
              "and make a decision.",
        "es": "¡Hola! Gracias por tu queja. Un administrador revisará las pruebas que enviaste y tomará una decisión.",
        "pt": "Olá! Obrigado pela sua denúncia. Um administrador vai analisar as provas enviadas e tomar uma decisão.",
    },
    # Та же жалоба без скриншотов и ссылок: не утверждаем, что доказательства приложены
    "complaint_no_evidence": {
        "ru": "Здравствуйте! Спасибо за жалобу. Администратор рассмотрит её и примет решение. Если у вас есть "
              "доказательства (скриншоты или видео), приложите их — без них проверить нарушение сложнее.",
        "kk": "Сәлеметсіз бе! Шағымыңыз үшін рақмет. Әкімші оны қарап, шешім қабылдайды. Дәлелдеріңіз (скриншот "
              "немесе видео) болса, оларды қоса тіркеңіз — онсыз бұзушылықты тексеру қиынырақ.",
        "uz": "Assalomu alaykum! Shikoyatingiz uchun rahmat. Administrator uni ko'rib chiqib, qaror qabul qiladi. "
              "Agar dalillaringiz (skrinshot yoki video) bo'lsa, ularni ilova qiling — ularsiz qoidabuzarlikni "
              "tekshirish qiyinroq.",
        "en": "Hello! Thank you for your report. An administrator will review it and make a decision. If you have "
              "evidence (screenshots or video), please attach it — without it the violation is harder to verify.",
        "es": "¡Hola! Gracias por tu queja. Un administrador la revisará y tomará una decisión. Si tienes pruebas "
              "(capturas de pantalla o vídeo), adjúntalas: sin ellas es más difícil comprobar la infracción.",
        "pt": "Olá! Obrigado pela sua denúncia. Um administrador vai analisá-la e tomar uma decisão. Se você tiver "
              "provas (capturas de tela ou vídeo), anexe-as — sem elas é mais difícil verificar a infração.",
    },
    # Апелляция: заявитель — наказанный игрок; не говорим ни «наказание верно», ни «снимем»
    "appeal_received": {
        "ru": "Здравствуйте! Ваше обращение по поводу наказания получено. Администратор повторно рассмотрит "
              "ситуацию и сообщит решение.",
        "kk": "Сәлеметсіз бе! Жаза туралы өтінішіңіз қабылданды. Әкімші жағдайды қайта қарап, шешімін хабарлайды.",
        "uz": "Assalomu alaykum! Jazo bo'yicha murojaatingiz qabul qilindi. Administrator vaziyatni qayta ko'rib "
              "chiqib, qarorini ma'lum qiladi.",
        "en": "Hello! Your appeal regarding the punishment has been received. An administrator will review "
              "the situation again and let you know the decision.",
        "es": "¡Hola! Hemos recibido tu apelación sobre la sanción. Un administrador revisará la situación de nuevo "
              "y te informará de la decisión.",
        "pt": "Olá! Recebemos o seu recurso sobre a punição. Um administrador vai analisar a situação novamente "
              "e informar a decisão.",
    },
    "received": {
        "ru": "Здравствуйте! Ваше обращение получено и будет рассмотрено администрацией.",
        "kk": "Сәлеметсіз бе! Өтінішіңіз қабылданды, оны әкімшілік қарастырады.",
        "uz": "Assalomu alaykum! Murojaatingiz qabul qilindi va ma'muriyat tomonidan ko'rib chiqiladi.",
        "en": "Hello! Your request has been received and will be reviewed by the administration.",
        "es": "¡Hola! Hemos recibido tu solicitud y la administración la revisará.",
        "pt": "Olá! Sua solicitação foi recebida e será analisada pela administração.",
    },
    "need_details": {
        "ru": "Здравствуйте! Пожалуйста, опишите проблему подробнее: что произошло, когда и с кем. "
              "Для жалобы приложите доказательства (скриншоты или видео).",
        "kk": "Сәлеметсіз бе! Мәселені толығырақ сипаттаңыз: не болды, қашан және кіммен. "
              "Шағым үшін дәлелдерді (скриншот немесе видео) қоса тіркеңіз.",
        "uz": "Assalomu alaykum! Iltimos, muammoni batafsil yozing: nima bo'ldi, qachon va kim bilan. "
              "Shikoyat uchun dalillarni (skrinshot yoki video) ilova qiling.",
        "en": "Hello! Please describe the problem in more detail: what happened, when and with whom. "
              "For a complaint, attach evidence (screenshots or video).",
        "es": "¡Hola! Por favor, describe el problema con más detalle: qué pasó, cuándo y con quién. "
              "Para una queja, adjunta pruebas (capturas de pantalla o vídeo).",
        "pt": "Olá! Por favor, descreva o problema com mais detalhes: o que aconteceu, quando e com quem. "
              "Para uma denúncia, anexe provas (capturas de tela ou vídeo).",
    },
    "duplicate": {
        "ru": "Здравствуйте! Похоже, это обращение повторяет предыдущее. Ответ будет дан в первом обращении.",
        "kk": "Сәлеметсіз бе! Бұл өтініш алдыңғысын қайталайтын сияқты. Жауап алғашқы өтінішке беріледі.",
        "uz": "Assalomu alaykum! Bu murojaat avvalgisini takrorlayotganga o'xshaydi. "
              "Javob birinchi murojaatda beriladi.",
        "en": "Hello! This request appears to repeat a previous one. You will receive an answer in your first request.",
        "es": "¡Hola! Parece que esta solicitud repite una anterior. Recibirás la respuesta en tu primera solicitud.",
        "pt": "Olá! Parece que esta solicitação repete uma anterior. Você receberá a resposta na sua primeira solicitação.",
    },
}


def template_reply(name: str, lang: str) -> str:
    return TEMPLATES[name].get(lang, TEMPLATES[name]["ru"])


# Жалобы на другого игрока (заявитель — пострадавший), в отличие от апелляций и пустых жалоб
COMPLAINT_ABOUT_OTHER = {
    "complaint.insult", "complaint.cheat", "complaint.dm", "complaint.scam",
    "complaint.rmt_ads", "complaint.other", "report.complaint",
}


def template_for(category: str, player_text: str | None = None, has_evidence: bool = True,
                 keyword_category: str | None = None) -> str:
    """Нейтральный шаблон, правильно адресованный заявителю.
    Апелляцию узнаём по тексту игрока: «не согласен с наказанием» может прийти и как report, и с категорией
    нарушения (keywords видят «оск» в «я не оскорблял»). complaint.admin — всегда апелляция: словарь
    апелляций знает не все языки, а нейтральный шаблон апелляции безопасен и для жалобы на админа."""
    appeal = bool(player_text and SENSITIVE_PATTERNS["appeal"][1].search(player_text))
    if category == "complaint.admin":
        # LLM назвала апелляцией жалобу пострадавшего («… забаньте его»):
        # ключевые слова видят нарушение другого игрока, признаков апелляции нет → благодарим за жалобу
        if keyword_category in COMPLAINT_ABOUT_OTHER and not appeal:
            category = keyword_category
        else:
            return "appeal_received"
    if category in COMPLAINT_ABOUT_OTHER:
        if appeal:
            return "appeal_received"
        return "complaint_received" if has_evidence else "complaint_no_evidence"
    return "received"
