# Matplotlib

The Matplotlib skill covers visualization patterns for CV/ML projects, including training curves, image grids, detection overlays, and publication-quality figures.

**Skill directory:** `skills/matplotlib/`

## Purpose

Visualization is essential for debugging models, presenting results, and understanding data. This skill teaches Claude Code to produce clean, consistent matplotlib figures following a unified style: proper axis labels, color palettes, figure sizing for papers and slides, and non-blocking display in training loops.

## When to Use

- Plotting training/validation loss and metric curves
- Visualizing image batches with labels or predictions
- Creating confusion matrices and precision-recall curves
- Generating publication-quality figures for papers
- Debugging data augmentation pipelines

## Key Patterns

### Training Curves

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: dict[str, list[float]],
    val_metrics: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation curves."""
    n_metrics = len(train_metrics) + 1  # +1 for loss
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    # Loss plot
    axes[0].plot(train_losses, label="Train", color="#1f77b4")
    axes[0].plot(val_losses, label="Val", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Metric plots
    for idx, (name, train_vals) in enumerate(train_metrics.items(), start=1):
        axes[idx].plot(train_vals, label="Train", color="#1f77b4")
        if name in val_metrics:
            axes[idx].plot(val_metrics[name], label="Val", color="#ff7f0e")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(name)
        axes[idx].set_title(name)
        axes[idx].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

### Image Grid

```python
def plot_image_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    ncols: int = 4,
    figsize_per_image: float = 3.0,
) -> plt.Figure:
    """Display a grid of images."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_image, nrows * figsize_per_image),
    )
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            ax.imshow(images[idx])
            if titles:
                ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    return fig
```

## Anti-Patterns to Avoid

- Do not call `plt.show()` in non-interactive contexts -- return the figure object or save to file
- Do not use default figure sizes -- always specify dimensions appropriate for the output medium
- Avoid the pyplot state machine for multi-figure scripts -- use the object-oriented API
- Do not forget `fig.tight_layout()` or `bbox_inches="tight"` -- prevents clipped labels

## Combines Well With

- **PyTorch Lightning** -- Log figures to experiment trackers via callbacks
- **OpenCV** -- Visualize processed images and detection results
- **TensorBoard** -- Embed matplotlib figures in TensorBoard summaries
- **W&B** -- Log figures as W&B images

## Full Reference

See [`skills/matplotlib/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/matplotlib/SKILL.md) for patterns including confusion matrix heatmaps, ROC curves, t-SNE embeddings, and animated training progress.
