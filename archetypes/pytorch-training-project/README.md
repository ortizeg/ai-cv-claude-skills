# PyTorch Training Project Archetype

A complete project template for training deep learning models with PyTorch Lightning. This archetype provides a production-ready scaffold that enforces best practices in experiment management, configuration, type safety, and reproducibility. It is designed for computer vision practitioners who need a structured, extensible foundation for training workflows ranging from simple classifiers to complex multi-task models.

## Purpose

The PyTorch Training Project archetype solves the recurring problem of bootstrapping ML training codebases from scratch. Every new training project faces the same set of infrastructure decisions: how to manage configurations, structure data loading, organize model definitions, handle logging, set up testing, and integrate CI/CD. This archetype encodes opinionated answers to all of those decisions so that you can focus on the modeling work itself.

The archetype is built around three pillars: PyTorch Lightning for structured training loops, Hydra for hierarchical configuration management, and Pydantic for runtime config validation. Together these ensure that experiments are reproducible, configurations are validated before training begins, and the training loop is cleanly separated from research logic.

## Use Cases

- **Image classification training** -- Train ResNet, EfficientNet, or custom backbones on labeled image datasets with built-in augmentation pipelines.
- **Object detection model development** -- Structure YOLO, Faster R-CNN, or SSD training with proper anchor generation and loss computation.
- **Semantic segmentation training** -- Build U-Net, DeepLab, or Mask2Former training pipelines with per-pixel loss functions and IoU metrics.
- **Transfer learning experiments** -- Fine-tune pretrained models on domain-specific data with frozen backbone scheduling and discriminative learning rates.
- **Hyperparameter optimization** -- Integrate with Optuna or Ray Tune through Hydra's sweeper plugins for systematic hyperparameter search.
- **Multi-GPU and distributed training** -- Leverage Lightning's built-in DDP, FSDP, and DeepSpeed strategies without modifying training code.

## Directory Structure

```
{{project_slug}}/
├── .github/
│   └── workflows/
│       ├── code-review.yml          # Automated PR code review
│       └── test.yml                 # CI test pipeline
├── .gitignore
├── .pre-commit-config.yaml          # Pre-commit hooks (ruff, mypy)
├── pixi.toml                        # Pixi package manager config
├── pyproject.toml                   # Project metadata and tool config
├── README.md                        # Generated project README
├── conf/                            # Hydra configuration directory
│   ├── config.yaml                  # Root config (composes groups)
│   ├── training/
│   │   ├── default.yaml             # Standard training settings
│   │   └── fast.yaml                # Quick iteration settings
│   ├── data/
│   │   └── default.yaml             # Data loading config
│   └── model/
│       └── default.yaml             # Model architecture config
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── py.typed                     # PEP 561 type marker
│   ├── configs/                     # Pydantic config models
│   │   ├── __init__.py
│   │   ├── training.py              # TrainingConfig dataclass
│   │   ├── data.py                  # DataConfig dataclass
│   │   └── model.py                 # ModelConfig dataclass
│   ├── data/                        # Data loading pipeline
│   │   ├── __init__.py
│   │   ├── datamodule.py            # LightningDataModule
│   │   ├── dataset.py               # Dataset implementations
│   │   └── transforms.py           # Albumentations pipelines
│   ├── models/                      # Model definitions
│   │   ├── __init__.py
│   │   ├── module.py                # LightningModule (train/val/test)
│   │   └── backbone.py             # Backbone network definitions
│   ├── callbacks/                   # Lightning callbacks
│   │   ├── __init__.py
│   │   └── visualization.py        # Prediction visualization
│   ├── metrics/                     # Custom metrics
│   │   ├── __init__.py
│   │   └── accuracy.py             # Task-specific metrics
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       └── io.py                    # File I/O helpers
├── scripts/
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Model evaluation script
│   └── export.py                    # ONNX/TorchScript export
├── notebooks/
│   └── exploration.ipynb            # Data exploration notebook
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Shared fixtures
    ├── unit/
    │   ├── test_model.py
    │   ├── test_data.py
    │   └── test_configs.py
    └── integration/
        └── test_training.py         # End-to-end training test
```

## Key Features

- **PyTorch Lightning** for structured, boilerplate-free training loops with automatic mixed precision, gradient accumulation, and distributed training.
- **Hydra** for hierarchical configuration management with command-line overrides, config composition, and multirun sweeps.
- **Pydantic** for strict runtime validation of all configuration values before training begins, catching typos and type errors early.
- **Full type safety** with mypy strict mode enabled by default and a `py.typed` marker for downstream consumers.
- **Pre-configured CI/CD** with GitHub Actions workflows for automated testing and code review.
- **Albumentations** integration for high-performance image augmentation pipelines.
- **Experiment tracking** integration points for Weights and Biases, MLflow, or TensorBoard (opt-in via Lightning loggers).

## Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `{{project_name}}` | Human-readable project name displayed in docs and logs | Required |
| `{{project_slug}}` | URL-safe directory name derived from the project name | Auto-generated |
| `{{package_name}}` | Python import name (underscored, PEP 8 compliant) | Auto-generated |
| `{{author_name}}` | Author full name for pyproject.toml | Required |
| `{{email}}` | Author email for pyproject.toml | Required |
| `{{description}}` | One-line project description | Required |
| `{{version}}` | Initial semantic version | 0.1.0 |
| `{{python_version}}` | Minimum Python version constraint | 3.11 |

## Dependencies

```toml
[dependencies]
python = ">=3.11"
pytorch = ">=2.0"
pytorch-lightning = ">=2.0"
hydra-core = ">=1.3"
pydantic = ">=2.0"
albumentations = ">=1.3"
torchmetrics = ">=1.0"
onnx = ">=1.14"
```

## Usage

### Project Initialization

```bash
# Start Claude and request project creation
claude
You: "Create a pytorch-training-project for object detection called 'yolo-detector'"

# Navigate into the generated project
cd yolo-detector

# Install all dependencies via pixi
pixi install
```

### Training

```bash
# Run training with default configuration
pixi run python scripts/train.py

# Override specific config values from the command line
pixi run python scripts/train.py training.learning_rate=1e-4 training.epochs=50

# Use a different config group preset
pixi run python scripts/train.py training=fast

# Run a Hydra multirun sweep
pixi run python scripts/train.py --multirun training.learning_rate=1e-3,1e-4,1e-5
```

### Evaluation and Export

```bash
# Evaluate a trained checkpoint
pixi run python scripts/evaluate.py checkpoint_path=outputs/best.ckpt

# Export to ONNX for deployment
pixi run python scripts/export.py checkpoint_path=outputs/best.ckpt export.format=onnx
```

### Testing

```bash
# Run the full test suite
pixi run pytest

# Run only unit tests
pixi run pytest tests/unit/

# Run with coverage
pixi run pytest --cov=src/
```

## Customization Guide

### Adding a New Model Architecture

1. Create a new module in `src/{{package_name}}/models/` (e.g., `efficientnet.py`).
2. Define a Pydantic config in `src/{{package_name}}/configs/model.py` with all architecture hyperparameters.
3. Add a corresponding Hydra YAML file in `conf/model/` (e.g., `efficientnet.yaml`) that sets `_target_` to your new class.
4. Write unit tests in `tests/unit/test_model.py` verifying forward pass shapes and gradient flow.
5. The training script will automatically discover the new model through Hydra's config group mechanism.

### Adding New Data Sources

1. Implement a `torch.utils.data.Dataset` subclass in `src/{{package_name}}/data/dataset.py`.
2. Add a Pydantic config in `src/{{package_name}}/configs/data.py` with paths, split ratios, and preprocessing options.
3. Create a Hydra YAML file in `conf/data/` referencing your new dataset class.
4. Update the `LightningDataModule` in `datamodule.py` to instantiate the new dataset.

### Adding Custom Callbacks

1. Create a new callback in `src/{{package_name}}/callbacks/` inheriting from `lightning.pytorch.Callback`.
2. Register it in the Hydra config under `config.yaml` or a dedicated `callbacks/` config group.
3. Common additions include early stopping schedulers, learning rate finders, and gradient norm loggers.

### Enabling Experiment Tracking

Lightning loggers can be added to the trainer by specifying them in the Hydra config or programmatically in `scripts/train.py`. Supported integrations include Weights and Biases (`WandbLogger`), MLflow (`MLFlowLogger`), and TensorBoard (`TensorBoardLogger`). Each logger is configured through its own Hydra config group for clean separation.
