# OpenCV Skill

## Purpose

This skill provides clean, type-safe abstractions over OpenCV's C-style Python API. It covers video reading/writing, image class wrappers with color space tracking, drawing utilities for bounding boxes and keypoints, camera capture, and conversion between NumPy and PyTorch formats.

## When to Use

- You need to read or write video files with proper resource management
- You are drawing detection results (bounding boxes, masks, keypoints) on images
- You want to avoid BGR/RGB color space bugs
- You need a clean camera capture abstraction
- You are converting images between OpenCV (numpy), PyTorch (tensor), and file formats

## Key Patterns

- **VideoReader ABC**: Abstract interface with OpenCV implementation; supports iteration, seeking, and metadata
- **Image class**: Wraps numpy arrays with tracked color space and safe conversion methods
- **Drawing utilities**: `draw_bounding_box`, `draw_detections`, `draw_keypoints`, `draw_mask_overlay`
- **Camera class**: Context-managed camera capture with configurable resolution and FPS
- **VideoWriter class**: Context-managed video output with codec configuration
- **Type conversions**: `numpy_to_torch`, `torch_to_numpy`, and batch conversion helpers

## Usage

```python
# Read video frames
with OpenCVVideoReader("video.mp4") as reader:
    for frame in reader:
        process(frame)

# Draw detections
annotated = draw_detections(image, boxes, labels, scores)

# Safe color space conversion
img = Image.from_file("photo.jpg")  # RGB
gray = img.to_color_space(ColorSpace.GRAY)
tensor = img.to_tensor()  # CHW float32
```

## Benefits

- Eliminates BGR/RGB confusion with explicit color space tracking
- Context managers guarantee resource cleanup for cameras and video files
- Abstract interfaces allow swapping OpenCV for alternative backends
- Type-safe drawing utilities with consistent API

## See Also

- `SKILL.md` in this directory for full documentation and code examples
- `matplotlib` skill for visualization and plotting
