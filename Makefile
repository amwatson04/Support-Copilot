# Support Copilot — основные команды.
# На Windows без make запускайте команды из правой части напрямую (см. раздел «Запуск» в README).

PYTHON ?= python
INPUT ?= data/tickets_valid.jsonl
OUTPUT ?= outputs/predictions_valid.jsonl
RPM ?= 12
TIMEOUT ?= 4

.PHONY: install run frontend test predict predict-offline measure

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn app.main:app --reload

# Интерфейс администратора (backend должен быть запущен: make run)
frontend:
	$(PYTHON) -m streamlit run frontend/app.py

test:
	$(PYTHON) -m pytest -q

# Обработка JSONL. Можно прервать и запустить снова — продолжит с места остановки.
predict:
	$(PYTHON) -m scripts.predict --input $(INPUT) --output $(OUTPUT) --rpm $(RPM) --timeout $(TIMEOUT)

# То же без LLM: быстро, данные никуда не отправляются
predict-offline:
	$(PYTHON) -m scripts.predict --input $(INPUT) --output outputs/predictions_offline.jsonl --provider disabled

measure:
	$(PYTHON) -m scripts.measure_llm --n 30
