# Common Patterns

Quick reference for the most frequently used patterns in AI/CV projects.

## Configuration Pattern

Use Pydantic BaseModel with Field validators for all configurations.

```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    """Validated training configuration."""
    learning_rate: float = Field(gt=0, default=1e-3, description="Learning rate")
    batch_size: int = Field(ge=1, default=32)
    epochs: int = Field(ge=1, default=100)

    model_config = {"frozen": True}

# Usage
config = TrainingConfig(learning_rate=0.01, batch_size=64)
```

**When to use:** Every time you define parameters that control behavior -- model hyperparameters, data pipeline settings, deployment configs.

---

## Abstraction Pattern

Wrap external libraries behind interfaces you control.

```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class ImageReader(ABC):
    """Abstract image reader interface."""

    @abstractmethod
    def read(self, path: Path) -> np.ndarray: ...

class OpenCVImageReader(ImageReader):
    """OpenCV implementation."""

    def read(self, path: Path) -> np.ndarray:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

**When to use:** Any external library (cv2, PIL, ffmpeg) used in business logic. Makes testing and swapping implementations easy.

---

## Testing Pattern

Use AAA (Arrange, Act, Assert) with synthetic CV data.

```python
import torch
import pytest

def test_model_output_shape():
    """Test model produces correct output shape."""
    # Arrange
    config = ModelConfig(num_classes=10)
    model = MyModel(config)
    image = torch.randn(2, 3, 224, 224)

    # Act
    output = model(image)

    # Assert
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

@pytest.mark.parametrize("size", [32, 64, 224])
def test_model_various_sizes(size: int):
    """Test model handles different input sizes."""
    model = MyModel(ModelConfig(num_classes=10))
    output = model(torch.randn(1, 3, size, size))
    assert output.shape[0] == 1
```

**When to use:** Every public API, model forward pass, data transformation, and utility function.

---

## Data Pipeline Pattern

LightningDataModule with Pydantic config and proper augmentations.

```python
import lightning as L
from torch.utils.data import DataLoader

class DataConfig(BaseModel):
    data_dir: str
    batch_size: int = Field(ge=1, default=32)
    num_workers: int = Field(ge=0, default=4)

class MyDataModule(L.LightningDataModule):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
```

**When to use:** Any training pipeline that loads and transforms data.

---

## Model Pattern

LightningModule with Pydantic config, proper logging, and optimizer setup.

```python
import lightning as L
import torch
from torch import nn

class MyModel(L.LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = self._build_model()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)
```

**When to use:** Every model training implementation.

---

## Inference Pattern

ONNX Runtime with Pydantic request/response schemas.

```python
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

class PredictionRequest(BaseModel):
    image_path: str
    confidence_threshold: float = Field(ge=0, le=1, default=0.5)

class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]

class PredictionResponse(BaseModel):
    detections: list[Detection]

class ONNXPredictor:
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path)

    def predict(self, image: np.ndarray) -> list[Detection]:
        outputs = self.session.run(None, {"input": image})
        return self._parse_outputs(outputs)
```

**When to use:** Production model serving, API endpoints, batch inference.

---

## Logging Pattern

Structured logging with proper levels -- never use `print()`.

```python
import logging

logger = logging.getLogger(__name__)

def train_epoch(epoch: int, loss: float) -> None:
    logger.info("Epoch completed", extra={"epoch": epoch, "loss": loss})

def load_model(path: str) -> None:
    logger.debug("Loading model from %s", path)
    try:
        model = torch.load(path)
    except FileNotFoundError:
        logger.error("Model file not found: %s", path)
        raise
```

**When to use:** Everywhere. Replace all `print()` calls with structured logging.

---

## Error Handling Pattern

Custom exceptions with Pydantic validation errors at boundaries.

```python
class ModelLoadError(Exception):
    """Raised when a model fails to load."""

class InvalidImageError(ValueError):
    """Raised when an image is invalid or corrupted."""

def load_image(path: Path) -> np.ndarray:
    """Load and validate an image file."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise InvalidImageError(f"Cannot decode image: {path}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"Expected 3-channel image, got shape: {image.shape}")

    return image
```

**When to use:** System boundaries (file I/O, API endpoints, user input). Don't over-validate internal function calls.
