.PHONY: install ingest serve eval test lint docker-up docker-down clean

install:
	pip install -r requirements.txt

ingest:
	python -m ingestion.loaders --source ./sample_docs

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	python -m evaluation.ragas_eval

test:
	pytest tests/ -v --asyncio-mode=auto

lint:
	ruff check . && ruff format --check .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

mlflow:
	mlflow ui --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache dist build *.egg-info
