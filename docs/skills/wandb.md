# Weights & Biases

The Weights & Biases skill covers experiment tracking, artifact management, and hyperparameter sweeps using W&B in CV/ML training pipelines.

**Skill directory:** `skills/wandb/`

## Purpose

Tracking experiments is critical for reproducible ML research and development. W&B provides a hosted dashboard for metrics, hyperparameters, model artifacts, and media (images, videos, 3D point clouds). This skill teaches Claude Code to integrate W&B deeply with PyTorch Lightning, log rich media for CV tasks, manage model artifacts with versioning, and configure hyperparameter sweeps.

## When to Use

- Training models where you need to compare runs across hyperparameters
- Projects that need artifact tracking for models, datasets, and predictions
- Team environments where experiment results must be shared and compared
- CV tasks where logging predictions as images or video is valuable

## Key Patterns

### Lightning Integration

```python
from __future__ import annotations

import lightning as L
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(
    project="object-detection",
    name="resnet50-baseline",
    log_model="all",      # Log checkpoints as artifacts
    save_dir="outputs/",
)

trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val/mAP",
            mode="max",
            save_top_k=3,
        ),
    ],
)
```

### Logging Images and Predictions

```python
import wandb

def log_predictions(
    images: list[np.ndarray],
    predictions: list[dict],
    ground_truths: list[dict],
    step: int,
) -> None:
    """Log prediction visualizations to W&B."""
    wandb_images = []
    for img, pred, gt in zip(images, predictions, ground_truths):
        wandb_img = wandb.Image(
            img,
            boxes={
                "predictions": {"box_data": pred["boxes"], "class_labels": pred["labels"]},
                "ground_truth": {"box_data": gt["boxes"], "class_labels": gt["labels"]},
            },
        )
        wandb_images.append(wandb_img)
    wandb.log({"predictions": wandb_images}, step=step)
```

### Sweep Configuration

```yaml
# sweep.yaml
program: src/my_project/train.py
method: bayes
metric:
  name: val/mAP
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  batch_size:
    values: [16, 32, 64]
  backbone:
    values: [resnet50, efficientnet_b0]
```

## Anti-Patterns to Avoid

- Do not log every single batch -- log at reasonable intervals to avoid rate limits
- Do not hardcode W&B project names -- use configuration or environment variables
- Avoid logging large tensors directly -- convert to images or scalars first
- Do not forget to call `wandb.finish()` in scripts (Lightning handles this automatically)

## Combines Well With

- **PyTorch Lightning** -- WandbLogger integrates seamlessly with Lightning Trainer
- **Hydra Config** -- Log resolved configs as run parameters
- **Docker CV** -- W&B API key as container environment variable
- **GitHub Actions** -- Optional training validation runs with W&B tracking

## Full Reference

See [`skills/wandb/SKILL.md`](https://github.com/ortizeg/ai-cv-claude-skills/blob/main/skills/wandb/SKILL.md) for patterns including custom W&B Tables for dataset visualization, model registry integration, and alert configuration for metric thresholds.
