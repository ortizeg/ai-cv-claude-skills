# PyTorch Lightning

The PyTorch Lightning skill provides expert patterns for building training pipelines with Lightning 2.x, covering `LightningModule`, `LightningDataModule`, callbacks, and trainer configuration.

**Skill directory:** `skills/pytorch-lightning/`

## Purpose

PyTorch Lightning abstracts the training boilerplate while keeping full control over the model and training logic. This skill encodes the best practices for structuring Lightning code in CV/ML projects: proper hook usage, metric computation patterns, checkpoint management, and multi-GPU training strategies.

## When to Use

Use this skill whenever your project involves model training with PyTorch. It is especially valuable for:

- Image classification, detection, or segmentation training pipelines
- Fine-tuning pretrained models
- Multi-GPU or multi-node distributed training
- Projects that need reproducible training with checkpointing

## Key Patterns

### LightningModule Structure

```python
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Accuracy

class ImageClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.head = nn.Linear(backbone.output_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["label"])
        self.train_acc(logits, batch["label"])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["label"])
        self.val_acc(logits, batch["label"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

### LightningDataModule

```python
class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 8) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ...
            self.val_dataset = ...

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
```

## Anti-Patterns to Avoid

- Do not call `.cuda()` or `.to(device)` manually -- Lightning handles device placement
- Do not use `torch.no_grad()` in validation steps -- Lightning applies it automatically
- Do not store batch data in `self` attributes -- it prevents proper memory cleanup
- Do not put data downloading logic in `__init__` -- use `prepare_data()` instead

## Combines Well With

- **Hydra Config** -- Structured configs for model, data, and trainer parameters
- **W&B / MLflow / TensorBoard** -- Logger integration for experiment tracking
- **Docker CV** -- Containerized training environments with GPU support
- **Testing** -- Unit tests for forward pass shapes and training step validation

## Full Reference

See [`skills/pytorch-lightning/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/pytorch-lightning/SKILL.md) for complete patterns including custom callbacks, profiling, and advanced distributed training configuration.
