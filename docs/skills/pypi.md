# PyPI

The PyPI skill covers Python package publishing conventions for CV/ML libraries, including pyproject.toml configuration, versioning, build systems, and release workflows.

**Skill directory:** `skills/pypi/`

## Purpose

When your ML code becomes a reusable library, it needs proper packaging. This skill teaches Claude Code to configure `pyproject.toml` for modern Python packaging, set up semantic versioning, define optional dependency groups (e.g., `[gpu]`, `[dev]`), and create automated release workflows. The result is a library that installs cleanly via `pip install your-package`.

## When to Use

- Building a reusable CV/ML library (model architectures, data loaders, utilities)
- Creating internal tools distributed via private PyPI
- Publishing open-source packages to PyPI
- Any project that other projects will depend on

## Key Patterns

### pyproject.toml Configuration

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-cv-lib"
version = "0.1.0"
description = "A computer vision utilities library"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "pillow>=10.0",
]

[project.optional-dependencies]
gpu = ["torch>=2.1", "torchvision>=0.16"]
dev = ["pytest>=7.0", "ruff>=0.1", "mypy>=1.5"]
docs = ["mkdocs-material>=9.0", "mkdocstrings[python]"]
all = ["my-cv-lib[gpu,dev,docs]"]

[project.urls]
Homepage = "https://github.com/you/my-cv-lib"
Documentation = "https://you.github.io/my-cv-lib"
Repository = "https://github.com/you/my-cv-lib"

[tool.hatch.build.targets.wheel]
packages = ["src/my_cv_lib"]
```

### Version Management

```python
# src/my_cv_lib/__init__.py
from __future__ import annotations

__version__ = "0.1.0"
```

### Build and Publish

```bash
# Build the package
uv run python -m build

# Check the distribution
uv run twine check dist/*

# Publish to PyPI
uv run twine upload dist/*
```

### Semantic Release Configuration

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
branch = "main"
build_command = "pip install build && python -m build"
upload_to_pypi = true
upload_to_release = true

[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

## Anti-Patterns to Avoid

- Do not use `setup.py` for new projects -- `pyproject.toml` is the standard
- Do not pin exact dependency versions in library code -- use compatible ranges
- Avoid including test files, notebooks, or data in the distribution
- Do not forget to define `__version__` in your package's `__init__.py`
- Do not use deprecated `version_variable` -- use `version_toml` for python-semantic-release v8+

## Combines Well With

- **Code Quality** -- Enforces quality standards on published code
- **Testing** -- Run tests before publishing
- **GitHub Actions** -- Automated release on tag push
- **Pre-commit** -- Clean code before it reaches the package

## Full Reference

See [`skills/pypi/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/pypi/SKILL.md) for patterns including namespace packages, C extension builds for CV operations, and TestPyPI workflows.
