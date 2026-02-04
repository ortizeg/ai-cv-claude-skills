# GitHub Actions Skill

CI/CD workflow patterns for ML/CV projects with tiered pipelines, GPU runner support, caching, and AI agent integration.

## Workflow Structure

Organize workflows into tiers based on speed and trigger frequency.

### Tier 1: Fast — Lint and Format (Every Push)

```yaml
# .github/workflows/lint.yml
name: Lint & Format

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true

      - name: Check formatting
        run: pixi run ruff format --check .

      - name: Lint
        run: pixi run ruff check .

      - name: Type check
        run: pixi run mypy src/
```

### Tier 2: Medium — Tests (Pull Requests)

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true

      - name: Run tests
        run: |
          pixi run pytest \
            --cov=src \
            --cov-report=xml \
            --cov-report=term \
            --cov-fail-under=80 \
            -v

      - name: Upload coverage
        if: always()
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

### Tier 3: Heavy — Training Validation (Merge to Main)

```yaml
# .github/workflows/train-validation.yml
name: Training Validation

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      epochs:
        description: "Number of epochs"
        required: false
        default: "2"

jobs:
  smoke-test:
    runs-on: self-hosted  # GPU runner
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true

      - name: Run smoke training
        run: |
          pixi run python -m my_project.train \
            trainer.max_epochs=${{ github.event.inputs.epochs || '2' }} \
            trainer.fast_dev_run=false \
            data.batch_size=4

      - name: Upload training artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: training-artifacts
          path: |
            outputs/
            checkpoints/
          retention-days: 7
```

## Matrix Testing

### Python Version + OS Matrix

```yaml
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.11", "3.12"]
        exclude:
          - os: macos-latest
            python-version: "3.12"
    runs-on: ${{ matrix.os }}
```

### CPU/GPU Matrix

```yaml
jobs:
  test:
    strategy:
      matrix:
        runner: [ubuntu-latest, self-hosted-gpu]
        include:
          - runner: ubuntu-latest
            device: cpu
          - runner: self-hosted-gpu
            device: cuda
    runs-on: ${{ matrix.runner }}
    steps:
      - name: Run tests
        run: pixi run pytest -v --device=${{ matrix.device }}
```

## Caching Strategies

### Pixi Environment Cache

```yaml
- name: Install pixi
  uses: prefix-dev/setup-pixi@v0.8.1
  with:
    pixi-version: latest
    cache: true  # Caches based on pixi.lock hash
```

### pip Cache (Fallback)

```yaml
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

### Model Weights Cache

```yaml
- name: Cache pretrained models
  uses: actions/cache@v4
  with:
    path: ~/.cache/torch/hub
    key: ${{ runner.os }}-torch-hub-${{ hashFiles('configs/model.yaml') }}
    restore-keys: |
      ${{ runner.os }}-torch-hub-
```

## GPU Runner Configuration

### Self-Hosted Runner Setup

```yaml
# Label self-hosted runners with GPU capabilities
jobs:
  train:
    runs-on: [self-hosted, gpu, linux]
    steps:
      - name: Verify GPU
        run: nvidia-smi

      - name: Set CUDA device
        run: echo "CUDA_VISIBLE_DEVICES=0" >> $GITHUB_ENV
```

### Runner Labels

| Label | Description |
|-------|-------------|
| `self-hosted` | Any self-hosted runner |
| `gpu` | Runner with NVIDIA GPU |
| `linux` | Linux OS |
| `a100` | Specific GPU type |

## Artifact Management

### Upload Model Checkpoints

```yaml
- name: Upload checkpoint
  uses: actions/upload-artifact@v4
  with:
    name: model-checkpoint-${{ github.sha }}
    path: checkpoints/best.ckpt
    retention-days: 30
```

### Upload Evaluation Reports

```yaml
- name: Generate evaluation report
  run: pixi run python -m my_project.evaluate --output=report.json

- name: Upload report
  uses: actions/upload-artifact@v4
  with:
    name: evaluation-report
    path: report.json
```

### Download Artifacts Across Jobs

```yaml
jobs:
  train:
    outputs:
      artifact-name: model-${{ github.sha }}
    steps:
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: model-${{ github.sha }}
          path: checkpoints/

  evaluate:
    needs: train
    steps:
      - name: Download model
        uses: actions/download-artifact@v4
        with:
          name: model-${{ github.sha }}
          path: checkpoints/
```

## Reusable Workflows

### Reusable Lint Workflow

```yaml
# .github/workflows/reusable-lint.yml
name: Reusable Lint

on:
  workflow_call:
    inputs:
      python-version:
        type: string
        default: "3.11"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true
      - run: pixi run lint
      - run: pixi run typecheck
```

### Calling Reusable Workflows

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint:
    uses: ./.github/workflows/reusable-lint.yml
    with:
      python-version: "3.11"

  test:
    needs: lint
    uses: ./.github/workflows/reusable-test.yml
```

## Secret Management

### Setting Secrets

```yaml
# Use GitHub repository secrets for API keys
steps:
  - name: Train with W&B logging
    env:
      WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    run: pixi run python -m my_project.train
```

### Environment Protection

```yaml
# Use environments for production deployments
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production  # Requires approval
    steps:
      - name: Deploy model
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
        run: pixi run deploy
```

## Agent Integration

### Code Review Agent

```yaml
# .github/workflows/code-review.yml
name: Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest

      - name: Run ruff format check
        run: pixi run ruff format --check .

      - name: Run ruff lint
        run: pixi run ruff check .

      - name: Run mypy
        run: pixi run mypy src/

      - name: Security checks
        run: pixi run ruff check --select S .
```

### Test Engineer Agent

```yaml
# .github/workflows/test-engineer.yml
name: Test Engineer

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest

      - name: Run tests with coverage
        run: |
          pixi run pytest \
            --cov=src \
            --cov-report=term \
            --cov-report=xml \
            --cov-fail-under=80

      - name: Check for skipped tests
        run: |
          if pixi run pytest --collect-only -q 2>&1 | grep -q "skipped"; then
            echo "::warning::Skipped tests found. Fix or remove them."
          fi
```

## Docker Build and Push

```yaml
# .github/workflows/docker.yml
name: Build Docker Image

on:
  push:
    branches: [main]
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          target: inference
```

## Semantic Release

Automate version bumps, changelog generation, and publishing with Python Semantic Release. The workflow runs after checks pass, analyzes commit messages since the last release, and determines the next version automatically.

### Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main, develop]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
          cache: true

      - name: Lint
        run: pixi run lint

      - name: Type check
        run: pixi run typecheck

      - name: Test
        run: pixi run test

  release:
    name: Semantic Release
    runs-on: ubuntu-latest
    concurrency: release
    environment: pypi
    needs: [check]
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop')
    permissions:
      id-token: write    # Required for OIDC / Trusted Publishing
      contents: write    # Required for creating releases/tags

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0   # Full history for commit analysis

      - name: Python Semantic Release
        id: release
        uses: python-semantic-release/python-semantic-release@v9.15.1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}

      - name: Publish to PyPI
        if: steps.release.outputs.released == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}

      - name: Publish to GitHub Releases
        if: steps.release.outputs.released == 'true'
        uses: python-semantic-release/upload-to-gh-release@v9.15.1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          tag: ${{ steps.release.outputs.tag }}
```

### Key Details

- **`fetch-depth: 0`** -- semantic release needs the full git history to analyze commits since the last tag
- **`concurrency: release`** -- prevents parallel release jobs from conflicting
- **`environment: pypi`** -- use a GitHub environment with deployment protection rules
- **`needs: [check]`** -- releases only run after lint, typecheck, and tests pass
- **Conditional publish** -- `steps.release.outputs.released == 'true'` only publishes when semantic release actually creates a new version
- **Trusted publishing** -- use `pypa/gh-action-pypi-publish` with OIDC (`id-token: write`) instead of long-lived API tokens when possible

### Branch Strategy

| Branch | Release Type | Version Example |
|--------|-------------|-----------------|
| `main` | Production release | `1.2.0` |
| `develop` | Prerelease | `1.3.0-dev.1` |

Configure in `pyproject.toml` (see PyPI skill for full config):

```toml
[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

## CI Verification (Required Before Task Completion)

A PR is **not done** until CI is green. Always verify CI after pushing:

```bash
# Check CI status on a PR
gh pr checks <PR_NUMBER>

# If a check fails, read the logs
gh run view <RUN_ID> --log-failed

# Fix the issue, commit, push, and re-check
```

### Common CI Failures and Fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| `ruff format --check` fails | Pre-commit ruff version differs from pixi ruff | Update `.pre-commit-config.yaml` rev to match `pixi run ruff -- --version` |
| `ruff check` fails | New lint violations | Run `pixi run lint` locally and fix |
| `mypy` fails | Type errors | Run `pixi run typecheck` locally and fix |
| `pytest` fails | Test failures | Run `pixi run test` locally and fix |
| Merge conflicts | Branch diverged from base | Merge/rebase base branch, resolve conflicts, re-run checks |

### Workflow

1. Run local checks: `pixi run format && pixi run lint && pixi run test`
2. Push and create PR
3. Run `gh pr checks <PR#>` -- wait for all checks to report
4. If any check fails: read logs, fix locally, push, repeat from step 3
5. Task is complete only when all checks are green

### Tool Version Alignment

Keep formatter/linter versions synchronized across all environments:

| Tool | Where | Must Match |
|------|-------|------------|
| ruff | `.pre-commit-config.yaml` rev | `pixi run ruff -- --version` |
| ruff | CI (`pixi run format-check`) | pixi.lock |
| mypy | CI (`pixi run typecheck`) | pixi.lock |

If pre-commit hooks reformat files that CI then rejects, the versions are out of sync. Update `.pre-commit-config.yaml` to match the pixi-managed version.

## Best Practices

1. **Use pixi in CI** -- keep CI commands identical to local development
2. **Cache aggressively** -- pixi environments, pip cache, model weights
3. **Fail fast on lint** -- run lint before tests to get quick feedback
4. **Pin action versions** -- use `@v4` not `@main` for stability
5. **Set timeouts** -- prevent runaway training jobs from consuming runner time
6. **Use artifacts wisely** -- set retention days, don't upload huge datasets
7. **Protect secrets** -- use GitHub Secrets, never hardcode credentials
8. **Use environments** -- require approval for production deployments
9. **Matrix strategically** -- test Python versions and OS combinations that matter
10. **Document runner requirements** -- label self-hosted runners clearly
11. **Verify CI after every push** -- never consider a task done until all checks pass
12. **Fix CI immediately** -- a broken CI blocks the entire team; treat failures as highest priority
