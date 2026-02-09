"""Training entry point for ${project_name}."""

from __future__ import annotations

import lightning as L
from loguru import logger


def main() -> None:
    """Run training."""
    from .data import DataConfig, ImageDataModule
    from .model import Classifier, ModelConfig

    model_config = ModelConfig()
    data_config = DataConfig()

    logger.info("Starting training: {}", model_config.backbone)

    model = Classifier(config=model_config)
    datamodule = ImageDataModule(config=data_config)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        precision="16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
