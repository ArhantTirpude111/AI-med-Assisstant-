# MedIntel AI — developer task runner
.DEFAULT_GOAL := help
.PHONY: help install install-full dev run cli test cov lint fmt type clean docker

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install core package (editable)
	pip install -e .

install-full:  ## Install with all optional + dev extras
	pip install -e ".[full,dev]"

dev: install-full  ## Alias for install-full

run:  ## Launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

cli:  ## Show CLI help
	python -m medintel.cli --help

test:  ## Run the test suite (offline)
	pytest

cov:  ## Run tests with coverage report
	pytest --cov=medintel --cov-report=term-missing

lint:  ## Lint with ruff
	ruff check src tests

fmt:  ## Auto-format with ruff
	ruff format src tests app

type:  ## Static type-check with mypy
	mypy src/medintel

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info

docker:  ## Build & run via docker compose
	docker compose -f docker/docker-compose.yml up --build
