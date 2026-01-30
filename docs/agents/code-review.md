# Code Review Agent

Automated code quality enforcement that runs as a GitHub Action. Must pass before any PR can be merged.

## Strictness Level

**Blocking** -- all checks must pass to merge.

## What It Checks

| Check | Tool | Blocking |
|-------|------|----------|
| Code formatting | Ruff format | Yes |
| Linting rules | Ruff check | Yes |
| Type safety | MyPy strict | Yes |
| Import sorting | Ruff (isort) | Yes |
| Security issues | Ruff (bandit) | Yes |

## GitHub Action Setup

Add to `.github/workflows/code-review.yml`:

```yaml
name: Code Review

on: [push, pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
      - run: pixi run ruff format --check .
      - run: pixi run ruff check .
      - run: pixi run mypy src/
```

## Local Development

Run checks locally before pushing to avoid CI failures:

```bash
# Run all checks
pixi run lint

# Auto-fix formatting
pixi run format

# Type check
pixi run typecheck
```

## Pre-commit Integration

Checks also run automatically on every commit via pre-commit hooks:

```bash
$ git commit -m "Add feature"
Ruff format..............................Passed
Ruff lint................................Passed
```

## Common Failures and Fixes

| Failure | Fix |
|---------|-----|
| `F401: imported but unused` | Remove unused import |
| `I001: import not sorted` | Run `pixi run format` |
| `S101: assert used` | Only use assert in test files |
| `T201: print found` | Use `logging` instead of `print` |
| `mypy: Missing type hints` | Add type annotations to all functions |
| `mypy: Incompatible types` | Fix type mismatch or add proper cast |

## Bypassing (Emergency Only)

```bash
# Skip pre-commit hooks locally
git commit --no-verify

# Note: CI will still block the PR merge!
```
