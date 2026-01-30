# Weights & Biases (W&B) Skill

## Purpose

This skill provides guidance on integrating Weights & Biases (wandb) for experiment tracking, metric visualization, artifact management, and hyperparameter optimization in machine learning projects. W&B is an opt-in dependency and all code should work without it installed.

## Usage

Reference this skill when:

- Setting up experiment tracking for a new ML project.
- Logging training metrics, images, or videos during model training.
- Managing model and dataset artifacts with versioning.
- Running hyperparameter sweeps with Bayesian optimization.
- Integrating W&B with PyTorch Lightning.
- Logging computer vision predictions (bounding boxes, segmentation masks).
- Promoting models through the W&B Model Registry.

## Opt-in Nature

W&B is never a hard requirement. All code should:

- Check for the `wandb` import and gracefully skip logging if unavailable.
- Accept a configuration flag (e.g., `use_wandb: bool`) to enable or disable tracking.
- Function correctly with no experiment tracking enabled.

## Setup

```bash
pip install wandb
wandb login
```

Set `WANDB_API_KEY` as an environment variable for CI/CD environments.

## What This Skill Covers

- Initialization and configuration with Pydantic models.
- Metric tracking (scalars, learning rates, gradients).
- Image logging with bounding boxes and segmentation masks.
- Video logging.
- Artifact management (models, datasets).
- Hyperparameter sweeps (Bayesian, grid, random).
- PyTorch Lightning integration via `WandbLogger`.
- Model Registry for production promotion.
- Offline mode and best practices.

## Benefits

- Centralized experiment tracking eliminates lost or forgotten configurations.
- Interactive dashboards make comparing runs effortless.
- Artifact versioning ensures full reproducibility.
- Sweep automation finds optimal hyperparameters efficiently.
- Team collaboration features support shared workspaces and reports.

See `SKILL.md` for complete documentation and code examples.
