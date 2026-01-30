# Abstraction Patterns Skill

You are writing well-abstracted Python code for AI/CV projects. Follow these patterns exactly.

## Core Philosophy

Abstraction exists to reduce cognitive load, not to add layers. In AI/CV projects, the right abstractions make code readable, testable, and reusable. The wrong abstractions make code harder to debug and impossible to understand. Every abstraction must justify its existence by making at least three call sites simpler.

## The Rule: When to Abstract

Abstract when:
- **Three or more call sites** share the same logic
- **Resource management** requires setup/teardown (video readers, model sessions, database connections)
- **Complex validation** needs to happen consistently (image format checking, bounding box validation)
- **External dependencies** need to be isolated for testing (file I/O, API calls, hardware access)

Do NOT abstract when:
- There is only one call site (inline it)
- The abstraction hides important details (GPU memory management, batch dimension handling)
- A simple function would suffice (do not create a class for a single method)

## Pattern 1: VideoReader Abstraction

Video reading involves resource management (opening/closing file handles), frame iteration, and metadata access. This is a perfect candidate for abstraction.

### The Problem Without Abstraction

```python
# BAD: Raw OpenCV video reading scattered across codebase
import cv2

cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # process frame...

cap.release()  # Easy to forget!
```

### The Abstraction

```python
"""Video reader with proper resource management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata for a video file.

    Attributes:
        path: Path to the video file.
        fps: Frames per second.
        total_frames: Total number of frames.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_seconds: Video duration in seconds.
    """

    path: Path
    fps: float
    total_frames: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Context-managed video reader using OpenCV.

    Provides safe resource management, frame iteration, and metadata access.
    Always use as a context manager to ensure resources are released.

    Args:
        path: Path to the video file.

    Example:
        with VideoReader("input.mp4") as reader:
            print(f"FPS: {reader.metadata.fps}")
            for frame in reader:
                process(frame)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata. Must be called after entering context."""
        if self._metadata is None:
            msg = "VideoReader must be used as a context manager"
            raise RuntimeError(msg)
        return self._metadata

    def __enter__(self) -> VideoReader:
        """Open the video file."""
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            msg = f"Failed to open video: {self._path}"
            raise RuntimeError(msg)

        self._metadata = VideoMetadata(
            path=self._path,
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return self

    def __exit__(self, *args: object) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames in the video."""
        if self._cap is None:
            msg = "VideoReader must be used as a context manager"
            raise RuntimeError(msg)

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a specific frame by index.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            Frame as a numpy array in BGR format.

        Raises:
            RuntimeError: If the frame cannot be read.
            IndexError: If frame_idx is out of bounds.
        """
        if self._cap is None:
            msg = "VideoReader must be used as a context manager"
            raise RuntimeError(msg)

        if frame_idx < 0 or frame_idx >= self.metadata.total_frames:
            msg = f"Frame index {frame_idx} out of range [0, {self.metadata.total_frames})"
            raise IndexError(msg)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            msg = f"Failed to read frame {frame_idx}"
            raise RuntimeError(msg)
        return frame
```

### Usage

```python
# Clean, safe, readable
with VideoReader("input.mp4") as reader:
    print(f"Video: {reader.metadata.fps} FPS, {reader.metadata.total_frames} frames")

    for frame in reader:
        detections = model.predict(frame)
        visualize(frame, detections)

    # Or read specific frames
    first_frame = reader.read_frame(0)
    middle_frame = reader.read_frame(reader.metadata.total_frames // 2)
```

## Pattern 2: Image Loading Abstraction

Image loading seems simple but involves format detection, color space conversion, validation, and error handling. Abstracting this prevents inconsistencies.

### The Abstraction

```python
"""Robust image loading with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ColorSpace = Literal["rgb", "bgr", "gray"]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
})


def load_image(
    path: str | Path,
    color_space: ColorSpace = "rgb",
    max_size: int | None = None,
) -> np.ndarray:
    """Load an image from disk with validation and optional resizing.

    Args:
        path: Path to the image file.
        color_space: Target color space ('rgb', 'bgr', or 'gray').
        max_size: If provided, resize so the longest edge is at most this value.

    Returns:
        Image as a numpy array in the requested color space.
        Shape is (H, W, 3) for color or (H, W) for grayscale.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file extension is not supported.
        RuntimeError: If the image cannot be decoded.
    """
    path = Path(path)

    if not path.exists():
        msg = f"Image not found: {path}"
        raise FileNotFoundError(msg)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        msg = f"Unsupported image format: {path.suffix}. Supported: {SUPPORTED_EXTENSIONS}"
        raise ValueError(msg)

    # Read image (OpenCV loads as BGR by default)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to decode image: {path}"
        raise RuntimeError(msg)

    # Convert color space
    if color_space == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # "bgr" requires no conversion

    # Optional resize
    if max_size is not None:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def save_image(
    image: np.ndarray,
    path: str | Path,
    color_space: ColorSpace = "rgb",
    quality: int = 95,
) -> None:
    """Save an image to disk.

    Args:
        image: Image array to save.
        path: Output file path.
        color_space: Color space of the input image.
        quality: JPEG quality (1-100). Only used for JPEG output.

    Raises:
        ValueError: If the image array is invalid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim not in (2, 3):
        msg = f"Expected 2D or 3D array, got {image.ndim}D"
        raise ValueError(msg)

    # Convert to BGR for OpenCV
    if color_space == "rgb" and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    params: list[int] = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    cv2.imwrite(str(path), image, params)
```

## Pattern 3: Metric Computation Abstraction

Metrics in CV projects often require accumulation over batches, thread safety, and reset semantics. Abstract this into a consistent interface.

### The Abstraction

```python
"""Metric computation with accumulation and reset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Metric(ABC):
    """Base class for accumulating metrics over batches.

    Subclasses must implement update() and compute().
    Call reset() between epochs.
    """

    @abstractmethod
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Update metric state with new predictions and targets."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute final metric values from accumulated state."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state for a new epoch."""
        ...


class AccuracyMetric(Metric):
    """Top-1 accuracy metric with accumulation.

    Example:
        metric = AccuracyMetric()
        for batch in dataloader:
            preds = model(batch.images)
            metric.update(preds, batch.labels)
        results = metric.compute()
        print(f"Accuracy: {results['accuracy']:.4f}")
        metric.reset()
    """

    def __init__(self) -> None:
        self._correct: int = 0
        self._total: int = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Update with batch predictions and targets."""
        pred_classes = np.argmax(predictions, axis=1)
        self._correct += int(np.sum(pred_classes == targets))
        self._total += len(targets)

    def compute(self) -> dict[str, float]:
        """Compute accuracy from accumulated counts."""
        if self._total == 0:
            return {"accuracy": 0.0}
        return {"accuracy": self._correct / self._total}

    def reset(self) -> None:
        """Reset counters."""
        self._correct = 0
        self._total = 0


class IoUMetric(Metric):
    """Intersection over Union metric for segmentation.

    Accumulates per-class IoU over batches and computes mean IoU.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Class index to ignore in computation.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._intersection = np.zeros(num_classes, dtype=np.int64)
        self._union = np.zeros(num_classes, dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Update with batch predictions and targets.

        Args:
            predictions: Predicted class indices, shape (N, H, W).
            targets: Ground truth class indices, shape (N, H, W).
        """
        mask = targets != self._ignore_index
        pred_masked = predictions[mask]
        target_masked = targets[mask]

        for cls in range(self._num_classes):
            pred_cls = pred_masked == cls
            target_cls = target_masked == cls
            self._intersection[cls] += int(np.sum(pred_cls & target_cls))
            self._union[cls] += int(np.sum(pred_cls | target_cls))

    def compute(self) -> dict[str, float]:
        """Compute per-class and mean IoU."""
        iou_per_class = np.zeros(self._num_classes)
        for cls in range(self._num_classes):
            if self._union[cls] > 0:
                iou_per_class[cls] = self._intersection[cls] / self._union[cls]

        result: dict[str, float] = {"mean_iou": float(np.mean(iou_per_class))}
        for cls in range(self._num_classes):
            result[f"iou_class_{cls}"] = float(iou_per_class[cls])
        return result

    def reset(self) -> None:
        """Reset accumulation arrays."""
        self._intersection[:] = 0
        self._union[:] = 0


class MetricCollection:
    """Collection of metrics computed together.

    Args:
        metrics: Dictionary mapping metric names to Metric instances.

    Example:
        metrics = MetricCollection({
            "accuracy": AccuracyMetric(),
            "iou": IoUMetric(num_classes=21),
        })
        for batch in dataloader:
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, metrics: dict[str, Metric]) -> None:
        self._metrics = metrics

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Update all metrics."""
        for metric in self._metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Compute all metrics and merge results."""
        results: dict[str, float] = {}
        for name, metric in self._metrics.items():
            for key, value in metric.compute().items():
                results[f"{name}/{key}"] = value
        return results

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self._metrics.values():
            metric.reset()
```

## Pattern 4: Model Inference Wrapper

Wrap model inference to handle preprocessing, batching, postprocessing, and device management in one place.

```python
"""Model inference wrapper with preprocessing and postprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class InferenceWrapper:
    """Wrapper for running model inference with pre/post processing.

    Handles device management, input preprocessing, batched inference,
    and output postprocessing in a single cohesive interface.

    Args:
        model: The PyTorch model.
        device: Device to run inference on.
        input_size: Expected input size (height, width).
        mean: Normalization mean per channel.
        std: Normalization std per channel.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        input_size: tuple[int, int] = (224, 224),
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self._model = model.to(device).eval()
        self._device = torch.device(device)
        self._input_size = input_size
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image for model input."""
        import cv2

        resized = cv2.resize(image, (self._input_size[1], self._input_size[0]))
        normalized = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image.

        Args:
            image: Input image as RGB numpy array.

        Returns:
            Model output as numpy array.
        """
        input_tensor = self.preprocess(image)
        output = self._model(input_tensor)
        return output.cpu().numpy()

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Run inference on a batch of images.

        Args:
            images: List of RGB numpy arrays.

        Returns:
            Batched model output as numpy array.
        """
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        output = self._model(batch)
        return output.cpu().numpy()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type[nn.Module],
        **kwargs: object,
    ) -> InferenceWrapper:
        """Load model from checkpoint and create wrapper."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = model_class(**checkpoint.get("hparams", {}))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model=model, **kwargs)
```

## Testing Abstractions

```python
"""Tests for abstraction patterns."""

from __future__ import annotations

import numpy as np
import pytest

from my_project.io import VideoReader, load_image
from my_project.metrics import AccuracyMetric, IoUMetric, MetricCollection


def test_accuracy_metric() -> None:
    """Test accuracy metric accumulation."""
    metric = AccuracyMetric()

    # Batch 1: 3/4 correct
    preds = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    targets = np.array([0, 1, 0, 0])
    metric.update(preds, targets)

    # Batch 2: 2/2 correct
    preds = np.array([[0.8, 0.2], [0.3, 0.7]])
    targets = np.array([0, 1])
    metric.update(preds, targets)

    result = metric.compute()
    assert result["accuracy"] == pytest.approx(5 / 6)


def test_accuracy_metric_reset() -> None:
    """Test that reset clears accumulated state."""
    metric = AccuracyMetric()
    preds = np.array([[0.9, 0.1]])
    targets = np.array([0])
    metric.update(preds, targets)
    metric.reset()
    result = metric.compute()
    assert result["accuracy"] == 0.0


def test_load_image_not_found() -> None:
    """Test that missing image raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image("/nonexistent/image.jpg")


def test_load_image_unsupported_format(tmp_path) -> None:
    """Test that unsupported format raises ValueError."""
    fake_file = tmp_path / "image.bpg"
    fake_file.touch()
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(fake_file)
```

## When NOT to Abstract

### Do Not Abstract Single-Use Logic

```python
# BAD: Unnecessary abstraction for one-off logic
class ImagePreprocessor:
    def __init__(self, size):
        self.size = size

    def process(self, image):
        return cv2.resize(image, self.size)

# GOOD: Just use the function directly
resized = cv2.resize(image, (224, 224))
```

### Do Not Hide Critical Details

```python
# BAD: Hides GPU memory management
class AutoBatcher:
    def auto_batch(self, items):
        # Magically figures out batch size based on GPU memory
        # Developer has no idea what's happening
        ...

# GOOD: Be explicit about batch size
for batch in DataLoader(dataset, batch_size=32):
    ...
```

### Do Not Create Classes for Single Functions

```python
# BAD: A class with one method is just a function
class NMSProcessor:
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, boxes, scores):
        return nms(boxes, scores, self.threshold)

# GOOD: Use functools.partial or just pass the argument
from functools import partial

apply_nms = partial(nms, iou_threshold=0.5)
filtered = apply_nms(boxes, scores)
```
