"""LightningDataModule for ${project_name}."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader


class DataConfig(BaseModel, frozen=True):
    """Data configuration."""

    data_dir: str = "data"
    batch_size: int = 32
    num_workers: int = 4
    num_classes: int = 10
    image_size: int = 224


class ImageDataModule(L.LightningDataModule):
    """Image dataset module.

    Args:
        config: Data configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        # Replace with your dataset loading logic
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if stage == "fit" or stage is None:
            self.train_dataset = datasets.FakeData(
                size=100, image_size=(3, self.config.image_size, self.config.image_size),
                num_classes=self.config.num_classes, transform=transform,
            )
            self.val_dataset = datasets.FakeData(
                size=20, image_size=(3, self.config.image_size, self.config.image_size),
                num_classes=self.config.num_classes, transform=transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
