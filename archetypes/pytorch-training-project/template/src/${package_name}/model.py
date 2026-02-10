"""LightningModule for ${project_name}."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from pydantic import BaseModel, Field
from torch import Tensor


class ModelConfig(BaseModel, frozen=True):
    """Model configuration."""

    num_classes: int = 10
    learning_rate: float = 1e-3
    backbone: str = "resnet18"
    pretrained: bool = True


class Classifier(L.LightningModule):
    """Image classification model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.backbone = self._build_backbone()
        self.head = nn.LazyLinear(config.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes,
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes,
        )

    def _build_backbone(self) -> nn.Module:
        """Build the backbone network."""
        import torchvision.models as models

        weights = "DEFAULT" if self.config.pretrained else None
        model = getattr(models, self.config.backbone)(weights=weights)
        # Remove the classification head
        model.fc = nn.Identity()
        return model

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

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
