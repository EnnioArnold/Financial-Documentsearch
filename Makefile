.PHONY: setup up down ingest test test-int test-e2e test-all eval kibana-setup lint fmt typecheck cov run
PY := .venv/bin/python
export PATH := $(CURDIR)/.venv/bin:$(PATH)

setup:  ## create venv + install deps
	uv venv --python 3.13 --allow-existing .venv
	uv pip install -e ".[dev]"
	@test -f .env || cp .env.example .env

up:  ## start Elasticsearch, Kibana, API+UI, import Kibana dashboard
	docker compose up -d --build --wait

down:  ## stop and remove containers (keeps volumes; add -v manually to wipe the index)
	docker compose down

ingest:  ## download + chunk + embed + index all sources (needs `make up` for ES)
	$(PY) -m finsearch.ingest.pipeline

test:  ## unit tests, no Elasticsearch needed
	$(PY) -m pytest

test-int:  ## integration tests against the docker ES
	$(PY) -m pytest -m integration -o addopts=""

test-e2e:  ## browser smoke test (Playwright chromium)
	$(PY) -m pytest -m e2e -o addopts=""

test-all: test test-int test-e2e

cov:  ## coverage of core logic
	$(PY) -m pytest --cov --cov-report=term-missing

eval:  ## run golden-set evaluation, writes reports/eval_results.md
	$(PY) -m finsearch.eval.runner

kibana-setup:
	$(PY) observability/setup_kibana.py

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

typecheck:
	$(PY) -m mypy src

run:  ## API + UI on the host (ES must be up)
	$(PY) -m uvicorn finsearch.api.main:app --reload --port 8000
