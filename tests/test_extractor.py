from app.services.extractor import extract_nickname, extract_rule_refs, extract_server, has_evidence
from app.services.rules import load_rules, rule_exists


def test_registry_has_prefixed_ids():
    rules = load_rules()
    assert "2.1" in rules        # общие правила без префикса
    assert "GZ 1.2" in rules     # зелёная зона
    assert "ЖБ 1.1.2" in rules   # правила жалоб
    assert not rule_exists("9.9")


def test_general_rules_spellings():
    assert extract_rule_refs("Игрок оскорблял меня, 2.1 ОП") == ["2.1"]
    assert extract_rule_refs("2.1оп") == ["2.1"]
    assert extract_rule_refs("5.2 OP, soft") == ["5.2"]
    assert extract_rule_refs("нарушил 2.1") == ["2.1"]
    assert extract_rule_refs("ОП 2.1") == ["2.1"]


def test_other_documents():
    assert extract_rule_refs("дм 1.2 GZ") == ["GZ 1.2"]
    assert extract_rule_refs("1.2 ГЗ") == ["GZ 1.2"]
    assert extract_rule_refs("ППЖ 1.1.2") == ["ЖБ 1.1.2"]
    assert extract_rule_refs("1.7.1 FW") == ["FW 1.7.1"]


def test_invalid_refs_are_dropped():
    assert extract_rule_refs("9.9 ОП") == []         # такого пункта нет
    assert extract_rule_refs("2.2 УК") == []         # уголовного кодекса нет в rules.json
    assert extract_rule_refs("было 10.10.2026") == []
    assert extract_rule_refs("жб") == []
    assert extract_rule_refs("") == []


def test_duplicates_removed():
    assert extract_rule_refs("2.1 ОП, ещё раз 2.1 и 5.2") == ["2.1", "5.2"]


def test_server():
    assert extract_server("пишу с сервера номер 18") == "18"
    assert extract_server("это было на 25 сервере вчера") == "25"
    assert extract_server("админ сервера 07.") == "7"
    assert extract_server("на 14-м сервере") == "14"
    assert extract_server("нарушил правила сервера 2.1 ОП") is None  # это пункт правил, а не сервер
    assert extract_server("оск родни") is None
    assert extract_server("Nick: Test_Player43\nServer: 2.1 OP") is None  # цифры из ника


def test_nickname_and_evidence():
    assert extract_nickname("ник Arman_Petrov12 оскорбил") == "Arman_Petrov12"
    assert extract_nickname("какой-то игрок") is None
    assert has_evidence("вот", attachments=2)
    assert has_evidence("фрапс: <url>", attachments=0)
    assert not has_evidence("скрины есть", attachments=0)  # слова игрока — не доказательство
