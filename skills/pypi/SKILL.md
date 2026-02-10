---
name: pypi
description: >
  Complete workflow for packaging and publishing Python projects to PyPI. Covers
  pyproject.toml configuration, src layout, version management, build system setup,
  trusted publishers via GitHub Actions, and TestPyPI staging.
---

# PyPI Publishing Skill

Complete workflow for packaging and publishing Python projects to PyPI. This skill covers `pyproject.toml` configuration, version management, build system setup, src layout, package metadata, entry points, dependency specification, building, publishing to PyPI and TestPyPI, trusted publishers via GitHub Actions, and release automation.

## Why Publish to PyPI

Publishing to PyPI makes your library installable with `pip install your-package` from anywhere. This is essential for:

- Sharing CV/ML utilities across projects and teams
- Distributing trained model inference wrappers
- Providing CLI tools for data processing pipelines
- Building an open-source community around your work
- Ensuring reproducible installations with pinned versions

## Project Layout

Use the `src` layout. It prevents accidental imports from the working directory (a common source of "works on my machine" bugs) and is the recommended standard.

```
my-cv-package/
├── src/
│   └── my_cv_package/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   └── classifier.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── transforms.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   └── visualization.py
│       └── cli.py
├── tests/
│   ├── conftest.py
│   ├── test_detector.py
│   └── test_transforms.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## pyproject.toml Configuration

### Using Hatchling (Recommended)

Hatchling is a modern, fast build backend with good defaults.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-cv-package"
version = "0.1.0"
description = "Computer vision utilities for object detection and classification"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "you@example.com"},
]
keywords = ["computer-vision", "deep-learning", "object-detection"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Typing :: Typed",
]

dependencies = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "torch>=2.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.4",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
all = ["my-cv-package[dev,docs]"]

[project.urls]
Homepage = "https://github.com/yourname/my-cv-package"
Documentation = "https://yourname.github.io/my-cv-package"
Repository = "https://github.com/yourname/my-cv-package"
Issues = "https://github.com/yourname/my-cv-package/issues"
Changelog = "https://github.com/yourname/my-cv-package/blob/main/CHANGELOG.md"

[project.scripts]
my-cv-tool = "my_cv_package.cli:main"

[project.entry-points."my_cv_package.models"]
resnet = "my_cv_package.models.classifier:ResNetClassifier"
yolo = "my_cv_package.models.detector:YOLODetector"
```

### Using Setuptools

If you prefer setuptools as the build backend:

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my-cv-package"
dynamic = ["version"]
description = "Computer vision utilities"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools_scm]
write_to = "src/my_cv_package/_version.py"
```

## Version Management

### Manual Versioning

Keep the version in `pyproject.toml` and `__init__.py`:

```python
# src/my_cv_package/__init__.py
"""My CV Package."""

__version__ = "0.1.0"
```

### Dynamic Versioning with setuptools-scm

Derive version from git tags automatically:

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
dynamic = ["version"]

[tool.setuptools_scm]
write_to = "src/my_cv_package/_version.py"
```

Then in `__init__.py`:

```python
try:
    from my_cv_package._version import version as __version__
except ImportError:
    __version__ = "0.0.0-dev"
```

Tag a release:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

### Dynamic Versioning with Hatch

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/my_cv_package/_version.py"
```

## Entry Points

### CLI Entry Points

Define command-line scripts that are installed with the package:

```toml
[project.scripts]
my-cv-detect = "my_cv_package.cli:detect_main"
my-cv-train = "my_cv_package.cli:train_main"
my-cv-export = "my_cv_package.cli:export_main"
```

```python
# src/my_cv_package/cli.py
import argparse


def detect_main() -> None:
    """CLI entry point for detection."""
    parser = argparse.ArgumentParser(description="Run object detection")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--model", default="yolov8n", help="Model name")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Output path")
    args = parser.parse_args()

    from my_cv_package.models.detector import detect
    detect(args.input, model=args.model, confidence=args.confidence, output=args.output)


def train_main() -> None:
    """CLI entry point for training."""
    ...


def export_main() -> None:
    """CLI entry point for model export."""
    ...
```

### Plugin Entry Points

Allow third-party extensions to register models:

```toml
[project.entry-points."my_cv_package.models"]
resnet = "my_cv_package.models.classifier:ResNetClassifier"
yolo = "my_cv_package.models.detector:YOLODetector"
```

Discover plugins at runtime:

```python
from importlib.metadata import entry_points


def get_available_models() -> dict[str, type]:
    """Discover all registered model plugins."""
    eps = entry_points(group="my_cv_package.models")
    return {ep.name: ep.load() for ep in eps}
```

## Building Packages

```bash
# Install build tool
pip install build

# Build source distribution and wheel
python -m build

# Output in dist/
# dist/my_cv_package-0.1.0.tar.gz    (sdist)
# dist/my_cv_package-0.1.0-py3-none-any.whl  (wheel)

# Inspect the wheel contents
unzip -l dist/my_cv_package-0.1.0-py3-none-any.whl
```

## Publishing to TestPyPI

Always test on TestPyPI first before publishing to the real PyPI.

```bash
# Install twine
pip install twine

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    my-cv-package
```

## Publishing to PyPI

```bash
# Upload to PyPI (requires API token)
twine upload dist/*

# Or with explicit token
twine upload -u __token__ -p pypi-YOUR-TOKEN-HERE dist/*
```

## Trusted Publisher (GitHub Actions)

The recommended approach is to use PyPI trusted publishers. This eliminates the need for API tokens by using OpenID Connect (OIDC) to authenticate GitHub Actions directly with PyPI.

### Setup on PyPI

1. Go to your PyPI project settings
2. Add a new trusted publisher
3. Enter your GitHub repository owner, name, workflow filename, and environment name

### Release Workflow

```yaml
# .github/workflows/release.yml
name: Release to PyPI

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write
  id-token: write  # Required for trusted publishing

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Needed for setuptools-scm

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install build dependencies
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  test-install:
    needs: build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Install wheel
        run: pip install dist/*.whl

      - name: Test import
        run: python -c "import my_cv_package; print(my_cv_package.__version__)"

  publish-testpypi:
    needs: test-install
    runs-on: ubuntu-latest
    environment: testpypi
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1

  github-release:
    needs: publish-pypi
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/*
          generate_release_notes: true
```

## Semantic Release

Python Semantic Release automates version bumps, changelogs, and publishing based on commit messages. It eliminates manual versioning and the release checklist below.

### Conventional Commits

Semantic release requires [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add YOLO v8 model loader          → bumps minor (0.1.0 → 0.2.0)
fix: correct NMS threshold handling      → bumps patch (0.2.0 → 0.2.1)
feat!: redesign detection pipeline API   → bumps major (0.2.1 → 1.0.0)

fix(data): handle empty annotation files
feat(models): add EfficientNet backbone
docs: update training guide
chore: update CI workflow
```

Only `feat` and `fix` (and breaking changes with `!` or `BREAKING CHANGE` footer) trigger releases. `docs`, `chore`, `ci`, `refactor`, `test`, and `style` do not.

### pyproject.toml Configuration

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
branch = "main"
build_command = "pip install build && python -m build"
upload_to_pypi = true
upload_to_release = true
commit_message = "chore(release): v{version} [skip ci]"
tag_format = "v{version}"

[tool.semantic_release.changelog]
template_dir = "templates"
changelog_file = "CHANGELOG.md"

[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

Key settings:

- `version_toml` -- points to the version field in `pyproject.toml` (replaces deprecated `version_variable`)
- `branch = "main"` -- production releases from `main`
- `branches.develop` -- prereleases (`0.3.0-dev.1`) from `develop`
- `build_command` -- runs before publishing; use your project's build backend
- `commit_message` -- includes `[skip ci]` to avoid infinite CI loops

### Using Flit as Build Backend

If your project uses Flit instead of Hatchling:

```toml
[build-system]
requires = ["flit_core>=3.4"]
build-backend = "flit_core.wheel"

[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "pip install flit && flit build"
upload_to_pypi = true
upload_to_release = true
```

### GitHub Actions Workflow

See the **GitHub Actions** skill for the full CI workflow that runs semantic release on push to `main` or `develop`.

```yaml
# .github/workflows/release.yml — summary
# Runs after checks pass, on push to main or develop
# Uses python-semantic-release/python-semantic-release@v9.15.1
# Publishes to PyPI via trusted publishers (OIDC)
# Creates GitHub releases with changelogs
```

### Version in __init__.py

Semantic release updates `pyproject.toml` automatically. Keep `__init__.py` in sync:

```python
# src/my_cv_package/__init__.py
"""My CV Package."""

from importlib.metadata import version

__version__ = version("my-cv-package")
```

This reads the version from the installed package metadata at runtime, so it always matches `pyproject.toml` without needing a separate update step.

## Release Checklist (Manual Alternative)

If not using semantic release, follow this process for every release:

```bash
# 1. Ensure all tests pass
pytest

# 2. Update version in pyproject.toml (if not using scm)
# Edit pyproject.toml: version = "0.2.0"

# 3. Update CHANGELOG.md
# Document what changed in this version

# 4. Commit the version bump
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.2.0"

# 5. Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# 6. Push commit and tag
git push origin main
git push origin v0.2.0
# GitHub Actions will build and publish automatically

# 7. Verify on PyPI
pip install --upgrade my-cv-package
python -c "import my_cv_package; print(my_cv_package.__version__)"
```

## Including Package Data

If your package includes non-Python files (model configs, default parameters):

```toml
# With hatchling
[tool.hatch.build.targets.wheel]
packages = ["src/my_cv_package"]

# Include data files
[tool.hatch.build.targets.wheel.force-include]
"configs" = "my_cv_package/configs"

# With setuptools
[tool.setuptools.package-data]
my_cv_package = ["configs/*.yaml", "configs/*.json"]
```

Access package data at runtime:

```python
from importlib import resources

def get_default_config() -> dict:
    """Load the default configuration shipped with the package."""
    config_file = resources.files("my_cv_package.configs").joinpath("default.yaml")
    with resources.as_file(config_file) as path:
        import yaml
        return yaml.safe_load(path.read_text())
```

## Best Practices

1. **Use src layout** -- Prevents accidental local imports during development.
2. **Use trusted publishers** -- No API tokens to manage or leak.
3. **Test on TestPyPI first** -- Always verify the package installs correctly before publishing to the real index.
4. **Pin build dependencies** -- Specify minimum versions for build tools.
5. **Use `opencv-python-headless`** -- The headless variant avoids GUI dependency conflicts in server environments.
6. **Declare all dependencies** -- Never assume packages are pre-installed. List everything in `dependencies`.
7. **Use optional dependencies** -- Put dev, docs, and heavy optional dependencies in `[project.optional-dependencies]`.
8. **Automate releases** -- Use GitHub Actions with tag-triggered workflows so publishing is a single `git tag` command.
9. **Include py.typed** -- Add an empty `py.typed` marker file to signal PEP 561 type stub support.
10. **Version with git tags** -- Use setuptools-scm or hatch-vcs to derive versions from tags, eliminating manual version bumps.
