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
	uv run pytest -v $(PYTEST_ARGS)

# Runs the suite with the blocking-fs detector installed: reports every
# filesystem call that happened on a tonio runtime worker. Opt-in because
# audit hooks cannot be uninstalled once added.
.PHONY: test-fs-detect
test-fs-detect:
	PIDREI_FS_DETECT=1 uv run pytest -q

.PHONY: release-check
release-check:
	uv run python scripts/release_check.py

.PHONY: models-data
models-data:
	uv run python packages/ai/scripts/generate_models.py

# Where the pi checkout lives; override with `make upstream-diff PI_ROOT=...`.
PI_ROOT ?= $(HOME)/Downloads/code/pi

.PHONY: upstream-diff
upstream-diff:
	uv run python scripts/upstream_diff.py --pi-root "$(PI_ROOT)"

.PHONY: upstream-bump
upstream-bump:
	uv run python scripts/upstream_diff.py --pi-root "$(PI_ROOT)" --bump "$(REF)"

.PHONY: all
all: format lint audit test
