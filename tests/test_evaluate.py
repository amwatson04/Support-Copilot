from scripts.evaluate import macro_f1, proxy_category


def test_proxy_category_from_admin_violation():
    assert proxy_category("2.1 ОП") == "complaint.insult"
    assert proxy_category("нарушение пункта 5.2 ОП") == "complaint.cheat"
    assert proxy_category("1.2 GZ") == "complaint.dm"
    assert proxy_category("ДМ") == "complaint.dm"
    assert proxy_category("крыто") is None
    assert proxy_category(None) is None


def test_macro_f1():
    assert macro_f1([("a", "a"), ("b", "b")]) == 1.0
    assert macro_f1([("a", "b"), ("b", "a")]) == 0.0
    assert 0 < macro_f1([("a", "a"), ("a", "b"), ("b", "b")]) < 1
