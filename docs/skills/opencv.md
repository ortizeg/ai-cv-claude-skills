# OpenCV

The OpenCV skill covers image processing, video handling, and classical computer vision operations using OpenCV with Python bindings.

**Skill directory:** `skills/opencv/`

## Purpose

OpenCV is the backbone of image manipulation in CV pipelines: loading images, color space conversions, geometric transforms, drawing annotations, and video I/O. This skill teaches Claude Code to use OpenCV correctly in the context of ML pipelines, including proper BGR/RGB handling, memory-efficient video processing, and thread-safe operations.

## When to Use

- Image preprocessing and augmentation pipelines
- Video frame extraction and processing
- Drawing bounding boxes, masks, and keypoints on images
- Classical CV operations (morphology, contour detection, homography)
- Camera capture and real-time processing

## Key Patterns

### Image Loading with Proper Color Handling

```python
from __future__ import annotations

import cv2
import numpy as np

def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 array.

    OpenCV loads as BGR by default. Always convert for ML pipelines.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image_rgb(image: np.ndarray, path: str | Path) -> None:
    """Save an RGB image to disk (converts to BGR for OpenCV)."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), bgr)
    if not success:
        raise IOError(f"Failed to save image: {path}")
```

### Drawing Detections

```python
def draw_detections(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
    scores: list[float],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes with labels on an image."""
    result = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        text = f"{label}: {score:.2f}"
        cv2.putText(result, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, color, 1, cv2.LINE_AA)
    return result
```

### Video Processing

```python
def process_video_frames(
    video_path: str | Path,
    process_fn: Callable[[np.ndarray], np.ndarray],
    output_path: str | Path | None = None,
) -> None:
    """Process each frame of a video with a given function."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_fn(frame)
            if writer:
                writer.write(processed)
    finally:
        cap.release()
        if writer:
            writer.release()
```

## Anti-Patterns to Avoid

- Never assume RGB ordering -- OpenCV uses BGR by default; always convert explicitly
- Do not ignore `cv2.imread` returning `None` -- it fails silently on missing files
- Avoid in-place modification of arrays when the original is needed downstream
- Do not forget to release `VideoCapture` and `VideoWriter` resources

## Combines Well With

- **PyTorch Lightning** -- Image preprocessing in DataModule pipelines
- **Matplotlib** -- Visualizing processed images and detection results
- **ONNX** -- Pre/post-processing around ONNX inference
- **Docker CV** -- OpenCV system dependencies in containers

## Full Reference

See [`skills/opencv/SKILL.md`](https://github.com/ortizeg/ai-cv-claude-skills/blob/main/skills/opencv/SKILL.md) for patterns including camera calibration, stereo vision, and high-performance image processing with CUDA-accelerated OpenCV.
