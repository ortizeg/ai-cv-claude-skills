# Code Quality

The Code Quality skill configures and enforces linting, formatting, and static type checking for CV/ML Python projects using ruff and mypy.

**Skill directory:** `skills/code-quality/`

## Purpose

ML code has a reputation for being messy -- inconsistent formatting, missing type hints, unused imports, and copy-pasted functions. This skill sets up a strict quality gate using ruff for linting and formatting, and mypy in strict mode for type checking. The result is ML code that meets the same standards as production software engineering.

## When to Use

Every project. Code quality is not optional. Load this skill alongside the Master Skill as part of your baseline configuration.

## Key Patterns

### Ruff Configuration

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific rules
    "D",    # pydocstyle
    "ANN",  # flake8-annotations
]
ignore = [
    "ANN101",  # missing type annotation for self
    "D100",    # missing docstring in public module
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.isort]
known-first-party = ["my_project"]
```

### Mypy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = ["torch.*", "torchvision.*", "albumentations.*", "cv2.*"]
ignore_missing_imports = true
```

### Running Quality Checks

```bash
# Format code
uv run ruff format src/ tests/

# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Type check
uv run mypy src/
```

## Anti-Patterns to Avoid

- Do not use `# type: ignore` without a specific error code -- always use `# type: ignore[specific-error]`
- Do not disable entire rule categories to make linting pass -- fix the code instead
- Do not skip type checking on ML code because "it is hard" -- the difficulty is a sign that the code needs better structure
- Avoid `noqa` comments without explanations

## Combines Well With

- **Master Skill** -- Provides the naming conventions ruff enforces
- **Pre-commit** -- Runs ruff and mypy automatically before commits
- **GitHub Actions** -- Runs quality checks in CI
- **Testing** -- Type-checked test code catches more bugs

## Full Reference

See [`skills/code-quality/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/code-quality/SKILL.md) for the full configuration including per-file rule overrides, custom ruff rules for ML patterns, and mypy plugin configuration for PyTorch and Pydantic.
