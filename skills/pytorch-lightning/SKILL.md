# PyTorch Lightning Skill

You are writing PyTorch Lightning code for AI/CV projects. Follow these patterns exactly.

## Core Philosophy

PyTorch Lightning separates research code (the model) from engineering code (training loops, distributed training, logging). Every training project in this framework uses Lightning as the standard training abstraction. Never write raw training loops.

## LightningModule Patterns

### Standard LightningModule Structure

Every model must follow this exact structure. The `LightningModule` encapsulates the model architecture, loss computation, optimizer configuration, and step logic.

```python
"""LightningModule for image classification."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor


class ImageClassifier(L.LightningModule):
    """Image classification model using Lightning.

    Args:
        num_classes: Number of output classes.
        learning_rate: Initial learning rate for optimizer.
        backbone: Name of the backbone architecture.
        pretrained: Whether to use pretrained weights.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = self._build_backbone(backbone, num_classes, pretrained)
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics - one instance per phase to avoid cross-contamination
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _build_backbone(
        self, backbone: str, num_classes: int, pretrained: bool
    ) -> nn.Module:
        """Build the backbone network."""
        import torchvision.models as models

        weights = "IMAGENET1K_V2" if pretrained else None
        model = getattr(models, backbone)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - only used for inference."""
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Shared computation for train/val/test steps."""
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }
```

### Key Rules for LightningModule

1. **Always call `self.save_hyperparameters()`** in `__init__` -- this enables automatic checkpoint loading and logging.
2. **Create separate metric instances** for train, val, and test to avoid state leakage between phases.
3. **Use `_shared_step`** to avoid duplicating forward logic across training_step, validation_step, and test_step.
4. **Log with namespaced keys** like `train/loss`, `val/acc` -- never use flat names like `loss` or `accuracy`.
5. **Return loss from `training_step`** -- Lightning uses it for backpropagation. Do not return anything from validation_step or test_step.
6. **Type all methods** with proper return annotations.

## LightningDataModule Patterns

### Standard DataModule Structure

Every dataset must be wrapped in a `LightningDataModule`. This separates data logic from model logic and ensures reproducibility.

```python
"""DataModule for image classification datasets."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageClassificationDataModule(L.LightningDataModule):
    """DataModule for image classification.

    Args:
        data_dir: Root directory containing train/val/test splits.
        batch_size: Batch size for dataloaders.
        num_workers: Number of dataloader workers.
        image_size: Target image size (height, width).
        pin_memory: Whether to pin memory for GPU transfer.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: tuple[int, int] = (224, 224),
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def train_transform(self) -> transforms.Compose:
        """Training augmentations."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def val_transform(self) -> transforms.Compose:
        """Validation/test transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(self.image_size[0] + 32),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                self.data_dir / "train",
                transform=self.train_transform,
            )
            self.val_dataset = ImageFolder(
                self.data_dir / "val",
                transform=self.val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                self.data_dir / "test",
                transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
```

### Key Rules for DataModule

1. **Always use `setup(stage)`** to initialize datasets -- never in `__init__`.
2. **Use `persistent_workers=True`** when `num_workers > 0` to avoid re-forking workers each epoch.
3. **Use `drop_last=True`** for training to avoid batch normalization issues with tiny final batches.
4. **Separate train and val transforms** -- validation must never include random augmentations.
5. **Use `pin_memory=True`** for GPU training to speed up host-to-device transfers.

## Trainer Configuration

### Standard Trainer Setup

```python
"""Training script using Lightning Trainer."""

from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger


def train() -> None:
    """Run training pipeline."""
    # Seed everything for reproducibility
    L.seed_everything(42, workers=True)

    # Initialize components
    datamodule = ImageClassificationDataModule(
        data_dir="data/imagenet",
        batch_size=64,
        num_workers=8,
    )

    model = ImageClassifier(
        num_classes=1000,
        learning_rate=1e-3,
        backbone="resnet50",
        pretrained=True,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch}-{val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # Logger
    logger = WandbLogger(
        project="image-classification",
        name="resnet50-baseline",
        log_model=True,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        val_check_interval=1.0,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test with best checkpoint
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
```

### Trainer Parameter Reference

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `accelerator` | `"auto"` | Auto-detect GPU/CPU/TPU |
| `devices` | `"auto"` | Use all available devices |
| `strategy` | `"auto"` | Auto-select DDP/FSDP/single |
| `precision` | `"16-mixed"` | Mixed precision for speed |
| `deterministic` | `True` | Reproducible results |
| `gradient_clip_val` | `1.0` | Prevent gradient explosion |
| `val_check_interval` | `1.0` | Validate every epoch |

## Callbacks

### Custom Callback Pattern

```python
"""Custom callbacks for training monitoring."""

from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback


class ImageLoggingCallback(Callback):
    """Log sample predictions as images to the logger.

    Args:
        num_samples: Number of samples to log per validation epoch.
    """

    def __init__(self, num_samples: int = 8) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log predictions on first batch of each validation epoch."""
        if batch_idx != 0:
            return

        images, labels = batch
        images = images[: self.num_samples]
        labels = labels[: self.num_samples]

        with torch.no_grad():
            logits = pl_module(images)
            preds = torch.argmax(logits, dim=1)

        # Log to wandb if available
        if hasattr(trainer.logger, "experiment"):
            import wandb

            trainer.logger.experiment.log({
                "val/predictions": [
                    wandb.Image(img, caption=f"pred={p}, true={t}")
                    for img, p, t in zip(images, preds, labels)
                ]
            })
```

### Frequently Used Built-in Callbacks

```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,      # Save best/last checkpoints
    EarlyStopping,        # Stop when metric plateaus
    LearningRateMonitor,  # Log learning rate to logger
    RichProgressBar,      # Better terminal progress bars
    StochasticWeightAveraging,  # SWA for better generalization
    GradientAccumulationScheduler,  # Variable accumulation
    ModelSummary,         # Print model architecture summary
)
```

## Logging

### Multi-Logger Setup

```python
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

loggers = [
    WandbLogger(project="my-project", name="experiment-1"),
    TensorBoardLogger(save_dir="logs/", name="my-project"),
    CSVLogger(save_dir="logs/", name="csv-logs"),
]

trainer = L.Trainer(logger=loggers)
```

### Logging Best Practices

```python
# In LightningModule methods:

# Scalar logging
self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

# Multiple scalars at once
self.log_dict({
    "val/loss": loss,
    "val/acc": acc,
    "val/f1": f1,
}, on_epoch=True)

# Image/artifact logging (use logger directly)
if self.logger:
    self.logger.experiment.log({"images": wandb_images})
```

## Distributed Training

### Multi-GPU with DDP

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed",
    sync_batchnorm=True,  # Important for multi-GPU with batch norm
)
```

### FSDP for Large Models

```python
from lightning.pytorch.strategies import FSDPStrategy

strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",
    activation_checkpointing_policy={nn.TransformerEncoderLayer},
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy=strategy,
    precision="16-mixed",
)
```

## Testing Lightning Code

```python
"""Tests for the image classifier module."""

from __future__ import annotations

import lightning as L
import pytest
import torch

from my_project.models.classifier import ImageClassifier


@pytest.fixture
def model() -> ImageClassifier:
    """Create a small model for testing."""
    return ImageClassifier(num_classes=10, backbone="resnet18", pretrained=False)


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy batch."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    return images, labels


def test_forward_shape(model: ImageClassifier, sample_batch: tuple) -> None:
    """Test that forward produces correct output shape."""
    images, _ = sample_batch
    output = model(images)
    assert output.shape == (4, 10)


def test_training_step_returns_loss(model: ImageClassifier, sample_batch: tuple) -> None:
    """Test that training_step returns a scalar loss."""
    # Lightning needs a trainer attached for logging
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, train_dataloaders=[sample_batch])


def test_save_and_load(model: ImageClassifier, tmp_path) -> None:
    """Test checkpoint save and load."""
    path = tmp_path / "model.ckpt"
    trainer = L.Trainer(default_root_dir=tmp_path, max_epochs=0)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)
    loaded = ImageClassifier.load_from_checkpoint(path)
    assert loaded.hparams.num_classes == 10
```

## Anti-Patterns to Avoid

1. **Never write raw training loops** -- always use Lightning Trainer.
2. **Never call `.cuda()` or `.to(device)`** -- Lightning handles device placement.
3. **Never call `optimizer.zero_grad()` or `optimizer.step()`** -- Lightning handles this.
4. **Never call `loss.backward()`** -- Lightning handles backpropagation.
5. **Never use `model.train()` or `model.eval()`** -- Lightning manages train/eval mode.
6. **Never manually sync metrics in DDP** -- use `torchmetrics` which handles sync automatically.
7. **Never put data downloads in `setup()`** -- use `prepare_data()` which runs on rank 0 only.
