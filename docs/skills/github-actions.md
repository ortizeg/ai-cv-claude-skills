# GitHub Actions

The GitHub Actions skill defines CI/CD workflow patterns for CV/ML projects, covering testing, linting, Docker builds, and model validation pipelines.

**Skill directory:** `skills/github-actions/`

## Purpose

ML projects need CI just as much as web applications -- perhaps more so, since silent numerical errors can go undetected. This skill teaches Claude Code to write GitHub Actions workflows that lint, type-check, test, build containers, and optionally validate model training on every pull request.

## When to Use

- Setting up CI for a new ML project
- Adding automated testing, linting, or type checking
- Automating Docker image builds and pushes
- Creating release workflows for Python packages or model artifacts

## Key Patterns

### Standard CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: v0.30.0
      - run: pixi run ruff check src/ tests/
      - run: pixi run ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
      - run: pixi run mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
      - run: pixi run pytest tests/ -v --tb=short
```

### Docker Build and Push

```yaml
# .github/workflows/docker.yml
name: Docker

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Semantic Release Workflow

```yaml
# .github/workflows/release.yml
release:
  name: Semantic Release
  runs-on: ubuntu-latest
  concurrency: release
  environment: pypi
  needs: [check]
  if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
  permissions:
    id-token: write
    contents: write
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    - uses: python-semantic-release/python-semantic-release@v9.15.1
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
```

## Anti-Patterns to Avoid

- Do not run full model training in CI -- use smoke tests with tiny datasets and 1-2 epochs
- Do not install dependencies with `pip install` when using Pixi -- use `setup-pixi` action
- Avoid hardcoding secrets in workflow files
- Do not skip CI for "small changes" -- silent breakage compounds

## Combines Well With

- **Code Quality** -- Run ruff and mypy in CI
- **Testing** -- Run pytest in CI
- **Docker CV** -- Build and push images on release
- **PyPI** -- Publish packages on tag push

## Full Reference

See [`skills/github-actions/SKILL.md`](https://github.com/ortizeg/ai-cv-claude-skills/blob/main/skills/github-actions/SKILL.md) for patterns including GPU-enabled runners, model artifact caching, and matrix builds across Python versions.
