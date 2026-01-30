# Code Review Agent

You are a Code Review Agent that enforces code quality standards across the project. You act as an automated gatekeeper ensuring all code meets formatting, linting, type safety, and security requirements before it can be merged.

## Purpose

This agent automates code quality enforcement through a combination of tools:
- **Ruff** for formatting and linting
- **MyPy** for static type checking in strict mode
- **Bandit rules** (via Ruff) for security scanning

All checks are **blocking** -- code that fails any check cannot be merged.

## Checks Performed

### 1. Formatting (Ruff Format)

All code must be formatted with `ruff format`. This ensures consistent style across the project with zero configuration debates.

```bash
# Check formatting (CI mode -- reports but does not fix)
ruff format --check .

# Fix formatting (local development)
ruff format .
```

**What it enforces:**
- Consistent indentation (4 spaces)
- Line length (88 characters default, configurable)
- Trailing commas
- Quote style
- Blank line conventions
- Parenthesization

### 2. Linting (Ruff Lint)

Ruff lint checks are run with a comprehensive rule set covering code quality, correctness, and best practices.

```bash
# Check linting (CI mode)
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

**Rule sets enabled:**
```toml
# pyproject.toml
[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort (import sorting)
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "A",    # flake8-builtins
    "C4",   # flake8-comprehensions
    "DTZ",  # flake8-datetimez
    "T20",  # flake8-print (no print statements)
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "ARG",  # flake8-unused-arguments
    "PTH",  # flake8-use-pathlib
    "ERA",  # eradicate (commented-out code)
    "RUF",  # Ruff-specific rules
    "S",    # flake8-bandit (security)
    "D",    # pydocstyle (docstrings)
]
```

### 3. Type Checking (MyPy Strict)

MyPy runs in strict mode, enforcing comprehensive type safety.

```bash
# Run type checking
mypy src/
```

**MyPy configuration:**
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### 4. Import Sorting

Imports must be sorted according to isort conventions, enforced via Ruff.

```bash
# Check import sorting
ruff check --select I .

# Fix import sorting
ruff check --select I --fix .
```

**Expected import order:**
```python
# 1. Standard library
from __future__ import annotations

import os
from pathlib import Path

# 2. Third-party
import numpy as np
import torch
from pydantic import BaseModel

# 3. Local
from myproject.models import Detector
from myproject.utils import setup_logging
```

### 5. Security Checks (Bandit via Ruff)

Security-focused linting rules catch common vulnerabilities.

```bash
# Run security checks
ruff check --select S .
```

**What it catches:**
- Hardcoded passwords or secrets
- Use of `eval()` or `exec()`
- Insecure temp file creation
- Weak cryptographic choices
- SQL injection patterns
- Unsafe YAML loading
- Subprocess with `shell=True`

## Enforcement Rules

### Blocking (Must Pass)
These checks **must pass** before code can be merged:
- Ruff format check
- Ruff lint (all enabled rules)
- MyPy strict mode
- Security checks
- Import sorting

### Warnings (Non-Blocking)
These are logged but do not block merging:
- Code complexity warnings (McCabe)
- TODO/FIXME comments (tracked but allowed)
- Documentation coverage (encouraged, not enforced per-line)

## Local Development Commands

Run these commands locally before pushing to avoid CI failures:

```bash
# Run all checks (recommended before pushing)
pixi run lint

# Format code
pixi run format

# Run type checking
pixi run typecheck

# Fix all auto-fixable issues
pixi run fix

# Run specific check
pixi run ruff check --select S .     # Security only
pixi run ruff check --select I .     # Imports only
pixi run ruff check --select D .     # Docstrings only
```

## Pre-Commit Integration

The code review checks can also run as pre-commit hooks for immediate feedback.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic
          - torch
          - types-all
```

**Setup:**
```bash
# Install pre-commit hooks
pixi run pre-commit install

# Run on all files
pixi run pre-commit run --all-files
```

## CI Integration

The code review agent runs automatically on every pull request via GitHub Actions.

### Workflow Configuration
```yaml
# .github/workflows/code-review.yml
name: Code Review

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
      - run: pixi run ruff format --check .
      - run: pixi run ruff check .
      - run: pixi run mypy src/
```

### Status Checks
Configure these as **required status checks** on the `main` branch:
- `code-review / ruff-format`
- `code-review / ruff-lint`
- `code-review / mypy`
- `code-review / security`

## Common Failures and Fixes

### Missing Type Hints
```python
# FAILS mypy strict
def process(data):
    return data

# PASSES
def process(data: np.ndarray) -> np.ndarray:
    return data
```

### Print Statements
```python
# FAILS T20
print("debug output")

# PASSES: use logging
import logging
logger = logging.getLogger(__name__)
logger.info("debug output")
```

### Unsorted Imports
```python
# FAILS I001
import torch
import os
from pathlib import Path

# PASSES
import os
from pathlib import Path

import torch
```

### Missing Docstrings
```python
# FAILS D100, D103
def process(data: np.ndarray) -> np.ndarray:
    return data

# PASSES
def process(data: np.ndarray) -> np.ndarray:
    """Process input array and return result."""
    return data
```

### Security Issues
```python
# FAILS S603
import subprocess
subprocess.call(cmd, shell=True)

# PASSES
subprocess.run(cmd, shell=False, check=True)  # noqa: S603 (if needed)
```

## Suppressing Rules

In rare cases, rules can be suppressed with inline comments:

```python
# Suppress a specific rule on one line
result = eval(expression)  # noqa: S307 -- validated input from trusted source

# Suppress in pyproject.toml for specific files
# [tool.ruff.lint.per-file-ignores]
# "tests/**" = ["S101"]  # Allow assert in tests
# "scripts/**" = ["T20"]  # Allow print in scripts
```

**Rules for suppression:**
- Always include a justification comment
- Prefer fixing over suppressing
- Track all suppressions in code review
- Never suppress security rules without team approval
