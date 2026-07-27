.DEFAULT_GOAL := all
pysources = packages

.PHONY: format
format:
	uv run ruff check --fix $(pysources)
	uv run ruff format $(pysources)

.PHONY: lint
lint:
	uv run ruff check $(pysources)
	uv run ruff format --check $(pysources)

.PHONY: audit
audit:
	uv run python scripts/audit.py

.PHONY: test
test:
	uv run pytest -v

.PHONY: release-check
release-check:
	uv run python scripts/release_check.py

.PHONY: models-data
models-data:
	uv run python packages/ai/scripts/generate_models.py

.PHONY: all
all: format lint audit test
