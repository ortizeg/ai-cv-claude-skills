# ${project_name}

${description}

## Setup

```bash
uv sync --all-extras
```

## Training

```bash
# Default training
uv run python -m ${package_name}.train

# With overrides
uv run python -m ${package_name}.train model=resnet50 trainer.max_epochs=50

# Debug run (1 batch)
uv run python -m ${package_name}.train trainer=debug
```

## Development

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format .
uv run mypy src/ --strict
```

## Project Structure

```
${project_slug}/
├── src/${package_name}/
│   ├── __init__.py
│   ├── model.py          # LightningModule
│   ├── data.py           # LightningDataModule
│   ├── train.py          # Training entry point
│   └── transforms.py     # Data augmentations
├── configs/
│   ├── config.yaml       # Main Hydra config
│   ├── model/
│   ├── data/
│   └── trainer/
├── tests/
├── pyproject.toml
└── README.md
```
