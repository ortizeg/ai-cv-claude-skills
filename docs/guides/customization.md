# Customization

This guide covers how to adapt the whet framework to your specific needs.

## Overriding Default Configurations

### Ruff Rules

The default Ruff configuration selects a broad set of rules. To customize:

```toml
# pyproject.toml
[tool.ruff]
line-length = 120  # Override default 100

select = [
    "E", "F", "I", "N", "UP", "S", "B", "A", "C4", "T20", "SIM",
    "PTH",  # Add: prefer pathlib over os.path
]

ignore = [
    "S101",  # Allow assert in tests
    "E501",  # Disable line length (handled by formatter)
]

[tool.ruff.per-file-ignores]
"tests/**/*.py" = ["S101", "T201"]
"scripts/**/*.py" = ["T201"]        # Allow print in scripts
"notebooks/**/*.py" = ["T201", "E402"]  # Allow print and late imports
```

### MyPy Settings

Adjust strictness or add overrides for specific packages:

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true

# Relax for specific packages without stubs
[[tool.mypy.overrides]]
module = ["cv2.*", "albumentations.*", "onnxruntime.*"]
ignore_missing_imports = true

# Relax for generated code
[[tool.mypy.overrides]]
module = ["my_project.generated.*"]
disallow_untyped_defs = false
```

### Coverage Thresholds

Adjust the minimum coverage:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-fail-under=70"  # Lower from 80 to 70
```

## Custom Pre-commit Hooks

Add hooks beyond the defaults:

```yaml
# .pre-commit-config.yaml
repos:
  # ... existing hooks ...

  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint
        args: [--ignore, DL3008]  # Ignore pin versions warning

  - repo: https://github.com/python-jsonschema/check-jsonschema
    rev: 0.28.0
    hooks:
      - id: check-github-workflows
```

## Extending Archetypes

### Adding Files to Templates

Add files to an archetype's `template/` directory:

```
archetypes/pytorch-training-project/template/
├── src/{{package_name}}/
│   └── callbacks/               # Add custom callback module
│       ├── __init__.py
│       └── visualization.py
└── scripts/
    └── export_model.py          # Add export script
```

### Creating Custom Archetypes

Create a new archetype by copying an existing one:

```bash
cp -r archetypes/pytorch-training-project archetypes/my-custom-archetype
```

Then modify the structure, README, and template files.

## Organization-Specific Variants

### Custom Skill Variants

Fork a skill to match your organization's standards:

```bash
cp -r skills/code-quality skills/my-org-code-quality
```

Edit `SKILL.md` to reflect organization-specific rules, naming conventions, or tooling requirements.

### Environment Variables

Projects can use `.env` files for local configuration:

```bash
# .env.example (committed to git)
WANDB_API_KEY=
AWS_DEFAULT_REGION=us-east-1
MODEL_CHECKPOINT_DIR=checkpoints/

# .env (gitignored, local only)
WANDB_API_KEY=my-secret-key
```

Load with:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    wandb_api_key: str
    model_checkpoint_dir: str = "checkpoints/"

    model_config = {"env_file": ".env"}
```

## Mono-Repo Setup

For organizations using mono-repos with multiple ML projects:

```
my-org-ml/
├── shared/                # Shared utilities
│   ├── pixi.toml
│   └── src/shared_utils/
├── projects/
│   ├── face-detection/    # Individual project
│   ├── pose-estimation/
│   └── depth-estimation/
├── pyproject.toml         # Root-level ruff/mypy config
└── .pre-commit-config.yaml
```

Tips for mono-repos:

- Keep a root `pyproject.toml` with shared tool configs
- Each project has its own `pixi.toml` for dependencies
- Use workspace-relative paths in all configurations
- Share common abstractions via the `shared/` package
- Run CI per-project using path filters in GitHub Actions
