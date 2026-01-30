# Matplotlib Skill

## Purpose

This skill provides reusable visualization patterns for ML and computer vision projects. It covers training curves, confusion matrices, image grids, bounding box overlays, feature map visualization, embedding plots (t-SNE/UMAP), publication-quality figures, and integration with experiment tracking tools.

## When to Use

- You need to plot training/validation loss and metric curves
- You want to visualize model predictions (detections, segmentations) on images
- You are preparing figures for a paper or presentation
- You need to display confusion matrices, feature maps, or embedding visualizations
- You want to log matplotlib figures to W&B, MLflow, or TensorBoard

## Key Patterns

- **Training curves**: Multi-panel loss and metric plots with log scale
- **Confusion matrix**: Normalized heatmap with cell annotations
- **Image grid**: Flexible grid layout for displaying augmentations or predictions
- **Detection overlay**: Bounding boxes with labels and scores using matplotlib patches
- **Feature maps**: Channel-wise visualization from CNN intermediate layers
- **Embeddings**: t-SNE and UMAP scatter plots with class-colored points
- **Publication style**: Preconfigured rcParams for journal/conference figures
- **Multi-format save**: Export to PNG, PDF, and SVG in one call

## Usage

```python
# Training curves
fig = plot_training_curves(train_losses, val_losses, save_path="curves.png")

# Confusion matrix
fig = plot_confusion_matrix(y_true, y_pred, class_names, save_path="cm.png")

# Image grid
plot_image_grid(images, titles=titles, ncols=4, save_path="grid.png")

# Log to experiment tracker
log_figure_to_wandb(fig, "eval/confusion_matrix", step=epoch)
```

## Benefits

- Reusable plotting functions eliminate boilerplate code
- Consistent styling across all project visualizations
- Publication-quality output with a single style setup call
- Clean integration with experiment tracking dashboards

## See Also

- `SKILL.md` in this directory for full documentation and code examples
- `opencv` skill for drawing directly on image arrays
