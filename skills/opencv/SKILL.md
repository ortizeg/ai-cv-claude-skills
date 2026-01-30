# OpenCV Skill

Comprehensive OpenCV abstractions for computer vision projects. This skill provides clean, type-safe wrappers around OpenCV's C-style API, covering video reading/writing, image abstractions, drawing utilities, color space conversions, and camera capture.

## Why Abstractions Over Raw OpenCV

OpenCV's Python API exposes the underlying C++ interface almost directly. While powerful, this leads to code that is error-prone and hard to maintain:

- Functions return magic integers (e.g., `cv2.CAP_PROP_FRAME_WIDTH`) instead of named properties
- Color channels are BGR by default, which surprises developers and causes subtle bugs
- No type hints on function signatures
- Resource management (releasing cameras, closing video writers) is manual
- Error handling is inconsistent (some functions return None, others raise)

The abstractions in this skill wrap OpenCV with Pythonic interfaces that are type-safe, context-managed, and consistent.

## VideoReader Abstraction

Define an abstract base class for video reading so you can swap implementations (OpenCV, FFmpeg, hardware-accelerated) without changing application code.

### Abstract Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable video metadata."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


class VideoReaderBase(ABC):
    """Abstract base class for video reading."""

    @abstractmethod
    def __init__(self, source: str | Path) -> None: ...

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        """Read the next frame. Returns None at end of video."""
        ...

    @abstractmethod
    def seek(self, frame_number: int) -> None:
        """Seek to a specific frame number."""
        ...

    @abstractmethod
    @property
    def metadata(self) -> VideoMetadata:
        """Return video metadata."""
        ...

    @abstractmethod
    def __enter__(self) -> "VideoReaderBase": ...

    @abstractmethod
    def __exit__(self, *args) -> None: ...

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def read_frames(self, start: int = 0, count: int | None = None) -> list[np.ndarray]:
        """Read a range of frames."""
        self.seek(start)
        frames = []
        for frame in self:
            frames.append(frame)
            if count is not None and len(frames) >= count:
                break
        return frames
```

### OpenCV Implementation

```python
import cv2
import numpy as np
from pathlib import Path


class OpenCVVideoReader(VideoReaderBase):
    """OpenCV-based video reader with context management."""

    def __init__(self, source: str | Path) -> None:
        self._path = Path(source)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")

        self._metadata = VideoMetadata(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=(
                int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                / max(self._cap.get(cv2.CAP_PROP_FPS), 1e-6)
            ),
            codec=self._decode_fourcc(
                int(self._cap.get(cv2.CAP_PROP_FOURCC))
            ),
        )

    @staticmethod
    def _decode_fourcc(fourcc: int) -> str:
        """Decode FourCC integer to string."""
        return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def read_frame(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def seek(self, frame_number: int) -> None:
        if frame_number < 0 or frame_number >= self._metadata.frame_count:
            raise ValueError(
                f"Frame {frame_number} out of range [0, {self._metadata.frame_count})"
            )
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def __enter__(self) -> "OpenCVVideoReader":
        return self

    def __exit__(self, *args) -> None:
        self._cap.release()

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
```

Usage:

```python
with OpenCVVideoReader("video.mp4") as reader:
    print(f"Resolution: {reader.metadata.resolution}")
    print(f"Duration: {reader.metadata.duration_seconds:.1f}s")

    for i, frame in enumerate(reader):
        # frame is RGB numpy array (H, W, 3)
        process_frame(frame)
        if i >= 100:
            break
```

## Image Class Abstraction

Wrap numpy arrays with metadata and safe conversion methods.

```python
from __future__ import annotations

from enum import Enum
from typing import Self

import cv2
import numpy as np


class ColorSpace(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"


class Image:
    """Type-safe image wrapper with color space tracking."""

    def __init__(self, data: np.ndarray, color_space: ColorSpace = ColorSpace.RGB) -> None:
        if data.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
        if data.ndim == 3 and data.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {data.shape[2]}")
        if data.ndim == 2 and color_space != ColorSpace.GRAY:
            raise ValueError("2D array must use GRAY color space")

        self._data = data
        self._color_space = color_space

    @classmethod
    def from_file(cls, path: str | Path, color_space: ColorSpace = ColorSpace.RGB) -> Self:
        """Load image from file."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        if color_space == ColorSpace.RGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cls(img, color_space=color_space if color_space != ColorSpace.RGB else ColorSpace.RGB)

    @classmethod
    def from_numpy(cls, array: np.ndarray, color_space: ColorSpace = ColorSpace.RGB) -> Self:
        """Create Image from numpy array."""
        return cls(array.copy(), color_space=color_space)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def color_space(self) -> ColorSpace:
        return self._color_space

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def channels(self) -> int:
        return self._data.shape[2] if self._data.ndim == 3 else 1

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def to_color_space(self, target: ColorSpace) -> Image:
        """Convert to a different color space."""
        if target == self._color_space:
            return self

        conversion_map = {
            (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
            (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
            (ColorSpace.RGB, ColorSpace.GRAY): cv2.COLOR_RGB2GRAY,
            (ColorSpace.BGR, ColorSpace.GRAY): cv2.COLOR_BGR2GRAY,
            (ColorSpace.GRAY, ColorSpace.RGB): cv2.COLOR_GRAY2RGB,
            (ColorSpace.GRAY, ColorSpace.BGR): cv2.COLOR_GRAY2BGR,
            (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
            (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
            (ColorSpace.RGB, ColorSpace.LAB): cv2.COLOR_RGB2LAB,
            (ColorSpace.LAB, ColorSpace.RGB): cv2.COLOR_LAB2RGB,
        }

        key = (self._color_space, target)
        if key not in conversion_map:
            raise ValueError(f"Unsupported conversion: {self._color_space} -> {target}")

        converted = cv2.cvtColor(self._data, conversion_map[key])
        return Image(converted, color_space=target)

    def resize(self, width: int, height: int, interpolation: int = cv2.INTER_LINEAR) -> Image:
        """Resize image."""
        resized = cv2.resize(self._data, (width, height), interpolation=interpolation)
        return Image(resized, color_space=self._color_space)

    def to_float32(self) -> Image:
        """Convert to float32 [0, 1] range."""
        if self._data.dtype == np.float32:
            return self
        return Image(self._data.astype(np.float32) / 255.0, color_space=self._color_space)

    def to_uint8(self) -> Image:
        """Convert to uint8 [0, 255] range."""
        if self._data.dtype == np.uint8:
            return self
        return Image((self._data * 255).clip(0, 255).astype(np.uint8), color_space=self._color_space)

    def to_tensor(self) -> "torch.Tensor":
        """Convert to PyTorch tensor (C, H, W) in float32."""
        import torch
        img = self.to_float32().to_color_space(ColorSpace.RGB)
        return torch.from_numpy(img.data.transpose(2, 0, 1))

    def save(self, path: str | Path) -> None:
        """Save image to file."""
        bgr = self.to_color_space(ColorSpace.BGR)
        cv2.imwrite(str(path), bgr.data)
```

## Drawing Utilities

Clean functions for drawing annotations on images.

```python
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Color:
    """RGB color."""
    r: int
    g: int
    b: int

    @property
    def bgr(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)


# Predefined colors
class Colors:
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    YELLOW = Color(255, 255, 0)
    CYAN = Color(0, 255, 255)
    MAGENTA = Color(255, 0, 255)
    WHITE = Color(255, 255, 255)
    BLACK = Color(0, 0, 0)

    PALETTE = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA]

    @classmethod
    def for_class(cls, class_id: int) -> Color:
        """Get a consistent color for a class ID."""
        return cls.PALETTE[class_id % len(cls.PALETTE)]


def draw_bounding_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str = "",
    color: Color = Colors.GREEN,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a bounding box with optional label on an image.

    Args:
        image: Image array (H, W, 3) in BGR format for OpenCV.
        box: Bounding box as (x1, y1, x2, y2).
        label: Text label to display above the box.
        color: Box and label color.
        thickness: Line thickness.
        font_scale: Font scale for the label.

    Returns:
        Image with drawn bounding box.
    """
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]

    cv2.rectangle(img, (x1, y1), (x2, y2), color.bgr, thickness)

    if label:
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            img, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color.bgr, -1
        )
        cv2.putText(
            img, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )

    return img


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    scores: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw multiple detection results on an image.

    Args:
        image: Image array (H, W, 3).
        boxes: Array of shape (N, 4) in xyxy format.
        labels: List of N label strings.
        scores: Optional array of N confidence scores.
        class_ids: Optional array of N class IDs for color assignment.
        thickness: Line thickness.

    Returns:
        Annotated image.
    """
    img = image.copy()
    for i in range(len(boxes)):
        color = Colors.for_class(class_ids[i] if class_ids is not None else i)
        text = labels[i]
        if scores is not None:
            text = f"{text} {scores[i]:.2f}"
        img = draw_bounding_box(img, tuple(boxes[i]), label=text, color=color, thickness=thickness)
    return img


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list[tuple[int, int]] | None = None,
    color: Color = Colors.GREEN,
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """Draw keypoints and optional skeleton connections.

    Args:
        image: Image array (H, W, 3).
        keypoints: Array of shape (N, 2) or (N, 3) with (x, y) or (x, y, confidence).
        skeleton: List of (start_idx, end_idx) pairs defining connections.
        color: Point color.
        radius: Point radius.
        thickness: Skeleton line thickness.

    Returns:
        Image with keypoints drawn.
    """
    img = image.copy()

    # Draw skeleton connections first (behind points)
    if skeleton is not None:
        for start, end in skeleton:
            pt1 = tuple(keypoints[start, :2].astype(int))
            pt2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(img, pt1, pt2, color.bgr, thickness)

    # Draw keypoints
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        conf = kp[2] if len(kp) > 2 else 1.0
        if conf > 0.5:
            cv2.circle(img, (x, y), radius, color.bgr, -1)

    return img


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Color = Colors.GREEN,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary mask on an image with transparency.

    Args:
        image: Image array (H, W, 3).
        mask: Binary mask (H, W) with values 0 or 1.
        color: Overlay color.
        alpha: Transparency (0 = invisible, 1 = opaque).

    Returns:
        Image with mask overlay.
    """
    img = image.copy()
    overlay = img.copy()
    overlay[mask.astype(bool)] = color.bgr
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
```

## Camera Capture Abstraction

```python
from contextlib import contextmanager

import cv2
import numpy as np


class Camera:
    """Context-managed camera capture."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read(self) -> np.ndarray:
        """Read a frame. Raises RuntimeError on failure."""
        if self._cap is None:
            raise RuntimeError("Camera not opened. Use 'with' statement or call open().")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __iter__(self):
        """Yield frames continuously."""
        while True:
            try:
                yield self.read()
            except RuntimeError:
                break
```

## Video Writer Abstraction

```python
class VideoWriter:
    """Context-managed video writer."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self._path}")
        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Write a frame (expects BGR format)."""
        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
```

Usage combining reader and writer:

```python
with OpenCVVideoReader("input.mp4") as reader:
    meta = reader.metadata
    with VideoWriter("output.mp4", meta.fps, meta.width, meta.height) as writer:
        for frame in reader:
            processed = process_frame(frame)
            # Convert RGB back to BGR for writer
            writer.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
```

## Integration with NumPy and PyTorch

```python
import numpy as np
import torch


def numpy_to_torch(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy array to CHW float32 tensor."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 tensor to HWC uint8 numpy array."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def batch_to_numpy(batch: torch.Tensor) -> list[np.ndarray]:
    """Convert BCHW tensor batch to list of HWC numpy arrays."""
    return [torch_to_numpy(batch[i]) for i in range(batch.shape[0])]
```

## Best Practices

1. **Always track color space** -- Use the `Image` class or explicit naming (`frame_rgb`, `frame_bgr`) to avoid silent BGR/RGB confusion.
2. **Use context managers** -- Always wrap `VideoCapture` and `VideoWriter` in `with` blocks to ensure resources are released.
3. **Convert to RGB early** -- Convert from BGR to RGB immediately after reading; convert back to BGR only when writing or displaying with OpenCV.
4. **Abstract over backends** -- Define interfaces (ABCs) so you can swap OpenCV for FFmpeg, Decord, or hardware decoders without changing application code.
5. **Validate inputs** -- Check that images have the expected dtype, shape, and value range before processing.
6. **Copy before mutating** -- OpenCV drawing functions modify arrays in place. Always `.copy()` if the original should be preserved.
7. **Use named constants** -- Define color palettes and codec strings as module-level constants, not magic values scattered through code.
