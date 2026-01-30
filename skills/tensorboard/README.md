# TensorBoard Logging Skill

## Purpose

This skill provides guidance on using TensorBoard for real-time visualization of training metrics, images, histograms, model graphs, and profiling data in PyTorch and PyTorch Lightning projects. TensorBoard is opt-in and requires no accounts, servers, or internet connection.

## Usage

Reference this skill when:

- Setting up lightweight experiment visualization for local development.
- Logging training and validation metrics (loss, accuracy, mAP).
- Visualizing model predictions, augmentation results, or feature maps.
- Monitoring weight and gradient distributions with histograms.
- Visualizing model computation graphs.
- Comparing hyperparameter configurations across runs.
- Profiling training performance to identify bottlenecks.
- Integrating logging with PyTorch Lightning.

## Opt-in Nature

TensorBoard is never a hard requirement. All code should:

- Check for the `tensorboard` import and gracefully skip logging if unavailable.
- Accept a configuration flag to enable or disable logging.
- Function correctly with no visualization enabled.

## Setup

```bash
pip install tensorboard
tensorboard --logdir=logs/ --port 6006
```

No account or API key required.

## What This Skill Covers

- Scalar logging (loss, metrics, learning rate).
- Image logging (individual images, grids, matplotlib figures).
- Histogram logging (weights, gradients, activations).
- Model graph visualization.
- Hyperparameter tuning with the HParams plugin.
- PyTorch Lightning integration (default logger).
- Custom summaries (text, embeddings, PR curves).
- Training profiling with PyTorch Profiler.
- Remote access via SSH tunnels and TensorBoard.dev.

## Benefits

- Zero infrastructure: no server, no account, no internet needed.
- Real-time visualization during training.
- Native PyTorch support through `torch.utils.tensorboard`.
- Minimal overhead on training performance.
- Built-in default logger for PyTorch Lightning.
- Rich visualization types beyond simple line charts.

See `SKILL.md` for complete documentation and code examples.
