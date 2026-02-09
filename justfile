# whet — development task runner

test:
    uv run pytest tests/ -v

test-cov:
    uv run pytest tests/ --cov=src/whet --cov-report=term --cov-report=html

lint:
    uv run ruff check .

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

typecheck:
    uv run mypy src/whet/ tests/ --strict

docs-serve:
    uv run mkdocs serve

docs-build:
    uv run mkdocs build

# Run all checks (lint + typecheck + test)
check: lint typecheck test

# Install whet in development mode
dev:
    uv sync --all-extras
