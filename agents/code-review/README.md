# Code Review Agent

Automated code quality enforcement via GitHub Actions.

## Strictness Level

**BLOCKING** — Must pass to merge

## What It Checks

1. **Formatting** — Ruff format
2. **Linting** — Ruff lint with full rule set
3. **Types** — MyPy strict mode
4. **Security** — Bandit rules via Ruff
5. **Imports** — Proper sorting

## Setup

Add to `.github/workflows/code-review.yml`

## Local Usage

```bash
pixi run lint
pixi run format
pixi run typecheck
```
