import pytest

from app.main import missing_data_files
from app.services import llm


def pytest_sessionstart(session):
    """Тестам нужны rules.json и taxonomy.json из материалов кейса — их нет в git.
    Без них — понятное сообщение вместо FileNotFoundError в середине прогона."""
    missing = missing_data_files()
    if missing:
        pytest.exit(f"Нет файлов кейса: {', '.join(missing)}. Положите их в папку data/ (см. README)", returncode=4)


@pytest.fixture(autouse=True)
def no_real_llm():
    """Тесты никогда не ходят в настоящую LLM, даже если ключ есть в .env."""
    llm.init({"LLM_PROVIDER": "disabled"})
    yield
    llm.init({"LLM_PROVIDER": "disabled"})
