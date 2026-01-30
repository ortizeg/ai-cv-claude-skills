# MLflow Tracking Skill

## Purpose

This skill provides guidance on integrating MLflow for experiment tracking, model management, and model serving in machine learning projects. MLflow is an open-source, self-hostable platform that manages the full ML lifecycle. It is opt-in and all code should work without it installed.

## Usage

Reference this skill when:

- Setting up experiment tracking with a self-hosted requirement.
- Logging parameters, metrics, and artifacts during training.
- Managing model versions through the MLflow Model Registry.
- Serving trained models as REST APIs.
- Integrating experiment tracking with PyTorch Lightning.
- Comparing MLflow with W&B for a project decision.
- Setting up a remote tracking server for team collaboration.

## Opt-in Nature

MLflow is never a hard requirement. All code should:

- Check for the `mlflow` import and gracefully skip logging if unavailable.
- Accept a configuration flag (e.g., `use_mlflow: bool`) to enable or disable tracking.
- Function correctly with no experiment tracking enabled.

## Setup

```bash
pip install mlflow
mlflow ui --port 5000
```

Set `MLFLOW_TRACKING_URI` for remote server usage.

## What This Skill Covers

- Local and remote tracking server setup.
- Experiment and run management.
- Parameter, metric, and tag logging.
- Artifact storage (files, figures, dictionaries).
- Model Registry with version and stage management.
- Model serving via REST API.
- PyTorch Lightning integration via `MLFlowLogger`.
- Comparison with W&B (feature matrix).
- Best practices for experiment organization.

## Benefits

- Fully open-source and self-hostable with no vendor lock-in.
- Built-in model serving capability.
- Standardized model packaging format.
- Centralized model registry with stage transitions.
- Works offline by default with local file-based storage.

See `SKILL.md` for complete documentation and code examples.
