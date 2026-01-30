# Archetypes

This directory contains 6 project templates for common AI/CV workflows.

## Available Archetypes

| Archetype | Use Case |
|-----------|----------|
| `pytorch-training-project` | Model training with Lightning, Hydra, experiment tracking |
| `cv-inference-service` | FastAPI + ONNX deployment service |
| `research-notebook` | Jupyter-based experimentation |
| `library-package` | Reusable Python package for PyPI |
| `data-processing-pipeline` | ETL workflows for datasets |
| `model-zoo` | Collection of pretrained models |

## Using an Archetype

```bash
claude

You: "Create a new project using the pytorch-training-project archetype"
```

The Master Skill will guide the initialization process.

## Common Structure

All archetypes share a base structure:

```
{{project_slug}}/
├── .github/workflows/
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml
├── pyproject.toml
├── README.md
├── src/{{package_name}}/
└── tests/
```
