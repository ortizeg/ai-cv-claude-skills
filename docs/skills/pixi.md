# Pixi

The Pixi skill covers modern Python package and environment management using Pixi, with a focus on CUDA-dependent CV/ML workflows.

**Skill directory:** `skills/pixi/`

## Purpose

ML projects have complex dependency chains: PyTorch requires specific CUDA versions, OpenCV needs system libraries, and ONNX Runtime has multiple backend variants. Pixi handles all of this with reproducible, lockfile-based environments that work across platforms. This skill teaches Claude Code to configure Pixi correctly for CV/ML projects, including GPU support, task runners, and multi-environment setups.

## When to Use

- Setting up a new project's dependency management
- Configuring CUDA-aware PyTorch installations
- Creating reproducible development environments
- Defining project tasks (train, test, lint, serve)

## Key Patterns

### Basic Project Configuration

```toml
# pixi.toml
[project]
name = "my-cv-project"
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.10,<3.13"
pytorch = ">=2.1"
torchvision = ">=0.16"
lightning = ">=2.1"
numpy = ">=1.24"
opencv = ">=4.8"
pillow = ">=10.0"

[pypi-dependencies]
albumentations = ">=1.3"
wandb = ">=0.16"

[tasks]
train = "python src/my_project/train.py"
test = "pytest tests/ -v"
lint = "ruff check src/ tests/"
format = "ruff format src/ tests/"
typecheck = "mypy src/"
```

### CUDA-Specific Configuration

```toml
[feature.cuda]
platforms = ["linux-64"]
channels = ["nvidia", "conda-forge", "pytorch"]

[feature.cuda.dependencies]
pytorch-cuda = "12.1.*"
cuda-toolkit = ">=12.1"

[environments]
default = { features = [], solve-group = "default" }
cuda = { features = ["cuda"], solve-group = "default" }
```

### Task Definitions

```toml
[tasks]
train = { cmd = "python src/my_project/train.py", description = "Run model training" }
test = { cmd = "pytest tests/ -v --tb=short", description = "Run test suite" }
serve = { cmd = "uvicorn src.my_project.serve:app --host 0.0.0.0 --port 8000" }
export = { cmd = "python scripts/export_onnx.py", depends-on = ["train"] }
```

## Anti-Patterns to Avoid

- Do not mix pip and conda installations for the same package -- use Pixi's `[pypi-dependencies]` section for PyPI-only packages
- Do not pin exact versions unless reproducing a specific bug -- use compatible ranges
- Do not use `requirements.txt` for project dependencies -- `pixi.toml` with lockfile is more reliable
- Avoid platform-specific dependencies in the main section -- use features for platform variants

## Combines Well With

- **Docker CV** -- Pixi installs inside Docker containers for reproducible builds
- **Code Quality** -- Pixi tasks for running ruff and mypy
- **GitHub Actions** -- Pixi environments in CI workflows
- **Testing** -- Pixi tasks for running the test suite

## Full Reference

See [`skills/pixi/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/pixi/SKILL.md) for advanced patterns including multi-platform builds, workspace configurations, and integration with conda-forge for system-level CV libraries.
