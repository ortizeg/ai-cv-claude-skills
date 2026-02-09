---
name: matplotlib
description: >
  Comprehensive visualization patterns for ML and computer vision using Matplotlib.
  Covers training curves, confusion matrices, image grids, bounding box overlays,
  feature maps, publication-quality figures, and experiment tracker integration.
---

# Matplotlib Skill

Comprehensive visualization patterns for ML and computer vision projects. This skill covers training curves, confusion matrices, image grids, bounding box overlays, feature maps, dimensionality reduction plots, publication-quality figures, and integration with experiment tracking.

## Why Matplotlib for ML

Matplotlib is the foundation of Python visualization. While higher-level libraries (Seaborn, Plotly) exist, matplotlib provides the control needed for:

- Precise figure layout for papers and presentations
- Custom annotations on images (bounding boxes, masks, keypoints)
- Multi-panel figures combining plots and images
- Consistent styling across an entire project
- Non-interactive rendering for servers and CI pipelines

## Style Configuration

Set up a consistent style at the project level. Create a style file or configure in code.

```python
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for servers (must be before pyplot import in scripts)
matplotlib.use("Agg")

# Project-wide style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "legend.fontsize": 10,
    "legend.framealpha": 0.8,
})
```

Or use a custom `.mplstyle` file:

```ini
# custom.mplstyle
figure.figsize: 10, 6
figure.dpi: 150
savefig.dpi: 300
savefig.bbox_inches: tight
font.size: 12
axes.grid: True
grid.alpha: 0.3
lines.linewidth: 2
```

Load with `plt.style.use("path/to/custom.mplstyle")`.

## Training Curve Plotting

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: dict[str, list[float]] | None = None,
    val_metrics: dict[str, list[float]] | None = None,
    save_path: str | Path | None = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot training and validation curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_metrics: Optional dict of training metrics {name: values}.
        val_metrics: Optional dict of validation metrics {name: values}.
        save_path: Optional path to save the figure.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    num_metrics = 1 + (len(train_metrics) if train_metrics else 0)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#FF5722")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Metric plots
    if train_metrics and val_metrics:
        for i, (name, train_vals) in enumerate(train_metrics.items(), 1):
            val_vals = val_metrics.get(name, [])
            axes[i].plot(epochs, train_vals, label=f"Train {name}", color="#2196F3")
            if val_vals:
                axes[i].plot(epochs, val_vals, label=f"Val {name}", color="#FF5722")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(name)
            axes[i].set_title(name)
            axes[i].legend()

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


# Usage
fig = plot_training_curves(
    train_losses=[2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2],
    val_losses=[2.6, 1.9, 1.4, 1.0, 0.7, 0.6, 0.55],
    train_metrics={"Accuracy": [0.3, 0.45, 0.6, 0.72, 0.83, 0.9, 0.94]},
    val_metrics={"Accuracy": [0.28, 0.42, 0.55, 0.65, 0.75, 0.78, 0.79]},
    save_path="training_curves.png",
)
```

## Confusion Matrix Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a confusion matrix with annotations.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        normalize: Whether to normalize by row (true class).
        save_path: Optional path to save.
        title: Figure title.
        cmap: Colormap name.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle classes with zero samples

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)

    return fig


# Usage
fig = plot_confusion_matrix(
    y_true=np.array([0, 0, 1, 1, 2, 2, 0, 1, 2]),
    y_pred=np.array([0, 1, 1, 1, 2, 0, 0, 1, 2]),
    class_names=["Cat", "Dog", "Bird"],
    save_path="confusion_matrix.png",
)
```

## Image Grid Display

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    ncols: int = 4,
    figsize_per_image: tuple[float, float] = (3, 3),
    save_path: str | Path | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Display a grid of images.

    Args:
        images: List of images as numpy arrays (H, W, 3) or (H, W) for grayscale.
        titles: Optional list of titles for each image.
        ncols: Number of columns.
        figsize_per_image: Size per image panel.
        save_path: Optional path to save.
        suptitle: Optional super title.

    Returns:
        Matplotlib figure.
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    figsize = (figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if idx < n:
            cmap = "gray" if images[idx].ndim == 2 else None
            ax.imshow(images[idx], cmap=cmap)
            if titles:
                ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


# Usage: display augmentation results
augmented = [augment(original_image) for _ in range(8)]
plot_image_grid(
    augmented,
    titles=[f"Aug {i}" for i in range(8)],
    ncols=4,
    save_path="augmentations.png",
    suptitle="Augmentation Examples",
)
```

## Bounding Box Overlay

```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    scores: np.ndarray | None = None,
    class_colors: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot detection results using matplotlib (better for notebooks/papers than OpenCV).

    Args:
        image: RGB image (H, W, 3).
        boxes: Bounding boxes (N, 4) in xyxy format.
        labels: List of N label strings.
        scores: Optional confidence scores.
        class_colors: Optional dict mapping class names to colors.
        save_path: Optional save path.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    default_colors = plt.cm.tab10.colors

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1

        if class_colors and label in class_colors:
            color = class_colors[label]
        else:
            color = default_colors[hash(label) % len(default_colors)]

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        text = label
        if scores is not None:
            text = f"{label} {scores[i]:.2f}"

        ax.text(
            x1, y1 - 5, text,
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig
```

## Feature Map Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_feature_maps(
    feature_map: torch.Tensor,
    num_channels: int = 16,
    ncols: int = 8,
    save_path: str | Path | None = None,
    title: str = "Feature Maps",
) -> plt.Figure:
    """Visualize feature map channels from a CNN layer.

    Args:
        feature_map: Tensor of shape (C, H, W) or (1, C, H, W).
        num_channels: Number of channels to display.
        ncols: Number of columns in the grid.
        save_path: Optional save path.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if feature_map.dim() == 4:
        feature_map = feature_map[0]  # Remove batch dimension

    fm = feature_map.detach().cpu().numpy()
    num_channels = min(num_channels, fm.shape[0])
    nrows = (num_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx < num_channels:
            ax.imshow(fm[idx], cmap="viridis")
            ax.set_title(f"Ch {idx}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


# Usage: hook into a model layer
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

model.layer3.register_forward_hook(hook_fn("layer3"))
model(input_batch)
plot_feature_maps(activations["layer3"], save_path="feature_maps.png")
```

## t-SNE / UMAP Plots

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    method: str = "tsne",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
    perplexity: int = 30,
) -> plt.Figure:
    """Visualize high-dimensional embeddings in 2D.

    Args:
        embeddings: Array of shape (N, D) with feature vectors.
        labels: Array of shape (N,) with integer class labels.
        class_names: Optional list of class name strings.
        method: 'tsne' or 'umap'.
        save_path: Optional save path.
        figsize: Figure size.
        title: Figure title.
        perplexity: t-SNE perplexity parameter.

    Returns:
        Matplotlib figure.
    """
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else str(label)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=name, s=10, alpha=0.7,
        )

    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{method.upper()} Embedding Visualization")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)

    return fig
```

## Publication-Quality Figures

```python
import matplotlib.pyplot as plt
import matplotlib


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.5),  # Single column width
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox_inches": "tight",
        "savefig.pad_inches": 0.05,

        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],

        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.5,

        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,

        "lines.linewidth": 1.0,
        "lines.markersize": 3,

        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",

        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,

        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
    })


def plot_comparison_bar_chart(
    methods: list[str],
    metrics: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing methods across metrics, suitable for papers.

    Args:
        methods: List of method names.
        metrics: Dict mapping metric names to lists of values per method.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    setup_publication_style()

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots()
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_metrics))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 100)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)

    return fig
```

## Saving Figures in Multiple Formats

```python
def save_figure_multiformat(
    fig: plt.Figure,
    base_path: str | Path,
    formats: list[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure in multiple formats.

    Args:
        fig: Matplotlib figure.
        base_path: Path without extension.
        formats: List of format strings.
        dpi: Resolution for raster formats.

    Returns:
        List of saved file paths.
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []

    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(path)

    return saved


# Usage
fig = plot_training_curves(train_losses, val_losses)
save_figure_multiformat(fig, "figures/training_curves", formats=["png", "pdf", "svg"])
```

## Integration with Experiment Tracking

```python
import matplotlib.pyplot as plt
import wandb


def log_figure_to_wandb(fig: plt.Figure, key: str, step: int | None = None) -> None:
    """Log a matplotlib figure to Weights & Biases."""
    wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_figure_to_mlflow(fig: plt.Figure, artifact_path: str) -> None:
    """Log a matplotlib figure to MLflow."""
    import mlflow
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


# Usage in training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if epoch % 10 == 0:
        fig = plot_training_curves(all_train_losses, all_val_losses)
        log_figure_to_wandb(fig, "training/curves", step=epoch)

        fig = plot_confusion_matrix(y_true, y_pred, class_names)
        log_figure_to_wandb(fig, "eval/confusion_matrix", step=epoch)
```

## Best Practices

1. **Always close figures** -- Call `plt.close(fig)` after saving to free memory, especially in training loops.
2. **Use the Agg backend** -- Set `matplotlib.use("Agg")` on servers to avoid display-related errors.
3. **Save as PDF for papers** -- Vector formats (PDF, SVG) scale without pixelation.
4. **Save as PNG for tracking** -- Raster formats render faster in dashboards.
5. **Use `tight_layout()`** -- Prevents labels from being cut off.
6. **Return figures** -- Functions should return `plt.Figure` so callers can further customize or save.
7. **Separate data from presentation** -- Compute metrics first, then pass arrays to plotting functions.
8. **Use consistent colors** -- Define a project color palette and use it across all plots.
9. **Label everything** -- Axes, legends, titles, and units. Unlabeled plots are useless.
10. **Use log scale for loss** -- Training loss curves are much more readable on a log scale.
