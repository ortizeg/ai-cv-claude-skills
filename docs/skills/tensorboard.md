# TensorBoard

The TensorBoard skill covers training visualization, model graph inspection, and profiling using TensorBoard in PyTorch-based CV/ML projects.

**Skill directory:** `skills/tensorboard/`

## Purpose

TensorBoard provides real-time training visualization without external services. This skill teaches Claude Code to log scalars, images, histograms, and model graphs to TensorBoard, configure the PyTorch profiler for performance analysis, and structure log directories for multi-experiment comparison. It is ideal for local development and research workflows that do not need cloud-hosted dashboards.

## When to Use

- Local development where you want quick metric visualization
- Research projects without cloud experiment tracker requirements
- Performance profiling of training and inference pipelines
- Model architecture visualization and weight distribution analysis

## Key Patterns

### Lightning Integration

```python
from __future__ import annotations

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir="outputs/",
    name="my-experiment",
    default_hp_metric=False,
)

trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
)
```

### Custom Image Logging

```python
from torch.utils.tensorboard import SummaryWriter

def log_image_grid(
    writer: SummaryWriter,
    images: Tensor,
    tag: str,
    step: int,
    normalize: bool = True,
) -> None:
    """Log a grid of images to TensorBoard."""
    from torchvision.utils import make_grid

    grid = make_grid(images, nrow=4, normalize=normalize)
    writer.add_image(tag, grid, global_step=step)
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("outputs/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        train_step(model, batch)
        prof.step()
```

## Anti-Patterns to Avoid

- Do not leave TensorBoard log directories unbounded -- add a cleanup policy or log rotation
- Do not log high-resolution images at every step -- subsample by step or epoch
- Avoid logging large histograms for every parameter at every step -- it bloats log size
- Do not use TensorBoard for experiment comparison across teams -- use W&B or MLflow for that

## Combines Well With

- **PyTorch Lightning** -- TensorBoardLogger is a first-class Lightning logger
- **Matplotlib** -- Log matplotlib figures to TensorBoard with `add_figure()`
- **Hydra Config** -- Log hyperparameters as TensorBoard hparams
- **Docker CV** -- Port-forward TensorBoard from training containers

## Full Reference

See [`skills/tensorboard/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/tensorboard/SKILL.md) for patterns including embedding visualization, custom scalar plugins, and multi-run comparison layouts.
