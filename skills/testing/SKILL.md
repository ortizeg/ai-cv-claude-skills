---
name: testing
description: >
  Comprehensive pytest patterns for ML and computer vision projects. Covers test
  structure, fixtures, parametrized tests, CV-specific assertions, tensor shape
  validation, mocking, performance testing, and CI integration.
---

# Testing Skill

Comprehensive pytest patterns for ML and computer vision projects. This skill covers test structure, fixtures, parametrized tests, CV-specific testing strategies, mocking, performance testing, and CI integration.

## Why Testing Matters in ML/CV

ML projects are especially prone to silent failures. A model can train without errors but produce garbage predictions due to incorrect preprocessing, wrong label mapping, transposed dimensions, or broken augmentation. Automated tests catch these issues before they waste hours of GPU time.

## Test Structure

Organize tests into three tiers: unit tests (fast, isolated), integration tests (component interactions), and end-to-end tests (full pipeline).

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_transforms.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_utils.py
├── integration/
│   ├── conftest.py
│   ├── test_training_step.py
│   ├── test_data_pipeline.py
│   └── test_inference.py
└── e2e/
    ├── conftest.py
    ├── test_train_eval.py
    └── test_export.py
```

## The AAA Pattern

Every test should follow Arrange-Act-Assert. This makes tests readable and maintainable.

```python
def test_model_forward_pass():
    """Test that model produces correct output shape."""
    # Arrange
    model = ResNet50(num_classes=10)
    batch = torch.randn(4, 3, 224, 224)

    # Act
    output = model(batch)

    # Assert
    assert output.shape == (4, 10)
    assert not torch.isnan(output).any()
```

## Fixtures and conftest.py

Use fixtures to create reusable test data and resources. Place shared fixtures in `conftest.py` at the appropriate level.

### Top-Level conftest.py

```python
# tests/conftest.py
import numpy as np
import pytest
import torch


@pytest.fixture
def seed():
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a synthetic RGB image (H, W, C) in uint8."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch(device) -> torch.Tensor:
    """Create a batch of normalized images (B, C, H, W)."""
    return torch.randn(4, 3, 224, 224, device=device)


@pytest.fixture
def sample_bboxes() -> np.ndarray:
    """Create sample bounding boxes in xyxy format."""
    return np.array([
        [10, 20, 100, 150],
        [200, 50, 350, 300],
        [50, 100, 200, 400],
    ], dtype=np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample class labels."""
    return np.array([0, 1, 2], dtype=np.int64)


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Create a small trained model for integration tests. Session-scoped for speed."""
    model = SmallTestModel(num_classes=5)
    # Minimal training to get non-random weights
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(10):
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 5, (2,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Create a temporary data directory with sample images."""
    for split in ["train", "val", "test"]:
        split_dir = tmp_path / split
        for cls in ["cat", "dog"]:
            cls_dir = split_dir / cls
            cls_dir.mkdir(parents=True)
            for i in range(5):
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(cls_dir / f"{i:03d}.jpg"), img)
    return tmp_path
```

### Unit Test conftest.py

```python
# tests/unit/conftest.py
import pytest
from myproject.models import SmallResNet, EfficientNetTiny


@pytest.fixture(params=["small_resnet", "efficientnet_tiny"])
def model_factory(request):
    """Parametrized fixture that yields different model architectures."""
    models = {
        "small_resnet": lambda nc: SmallResNet(num_classes=nc),
        "efficientnet_tiny": lambda nc: EfficientNetTiny(num_classes=nc),
    }
    return models[request.param]
```

## Parametrized Tests

Use `@pytest.mark.parametrize` to test multiple inputs without duplicating test code.

```python
import pytest
import torch


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("image_size", [224, 320, 640])
def test_model_handles_various_sizes(batch_size, image_size):
    """Model should handle different batch sizes and resolutions."""
    model = MyModel(num_classes=10)
    x = torch.randn(batch_size, 3, image_size, image_size)
    output = model(x)
    assert output.shape == (batch_size, 10)


@pytest.mark.parametrize(
    "bbox_format,expected",
    [
        ("xyxy", [10, 20, 110, 220]),
        ("xywh", [10, 20, 100, 200]),
        ("cxcywh", [60, 120, 100, 200]),
    ],
)
def test_bbox_conversion(bbox_format, expected):
    """Test bounding box format conversions."""
    bbox_xyxy = [10, 20, 110, 220]
    result = convert_bbox(bbox_xyxy, from_format="xyxy", to_format=bbox_format)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "num_classes,input_shape",
    [
        (10, (1, 3, 224, 224)),
        (100, (4, 3, 224, 224)),
        (1000, (1, 3, 384, 384)),
    ],
    ids=["10cls-single", "100cls-batch", "1000cls-highres"],
)
def test_classifier_output(num_classes, input_shape):
    """Test classifier with various class counts and inputs."""
    model = Classifier(num_classes=num_classes)
    x = torch.randn(*input_shape)
    out = model(x)
    assert out.shape == (input_shape[0], num_classes)
```

## CV-Specific Testing

### Testing with Synthetic Data

Never rely on real datasets in unit tests. Generate synthetic data that exercises the same code paths.

```python
import cv2
import numpy as np
import pytest


def make_synthetic_detection_sample(
    image_size: tuple[int, int] = (480, 640),
    num_objects: int = 5,
    num_classes: int = 10,
) -> dict:
    """Create a synthetic detection sample with image, boxes, and labels."""
    h, w = image_size
    image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    boxes = []
    for _ in range(num_objects):
        x1 = np.random.randint(0, w - 50)
        y1 = np.random.randint(0, h - 50)
        x2 = np.random.randint(x1 + 10, min(x1 + 200, w))
        y2 = np.random.randint(y1 + 10, min(y1 + 200, h))
        boxes.append([x1, y1, x2, y2])

    return {
        "image": image,
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.random.randint(0, num_classes, num_objects),
    }


def test_detection_dataset_returns_correct_types():
    """Verify dataset item structure and types."""
    sample = make_synthetic_detection_sample()
    assert sample["image"].dtype == np.uint8
    assert sample["boxes"].dtype == np.float32
    assert sample["boxes"].shape[1] == 4
    assert len(sample["labels"]) == len(sample["boxes"])
```

### Testing Augmentations

```python
import albumentations as A
import numpy as np
import pytest


@pytest.fixture
def augmentation_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.Resize(256, 256),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def test_augmentation_preserves_bbox_count(augmentation_pipeline):
    """Augmentation should not drop bounding boxes."""
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = [[10, 20, 100, 150], [200, 50, 350, 300]]
    labels = [0, 1]

    result = augmentation_pipeline(image=image, bboxes=bboxes, labels=labels)

    assert len(result["bboxes"]) == 2
    assert result["image"].shape == (256, 256, 3)


def test_horizontal_flip_mirrors_bboxes():
    """Horizontal flip should mirror bbox x-coordinates."""
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    bboxes = [[10, 20, 50, 80]]
    labels = [0]

    result = transform(image=image, bboxes=bboxes, labels=labels)

    # x1 should become width - original_x2 = 200 - 50 = 150
    assert result["bboxes"][0][0] == pytest.approx(150, abs=1)


def test_augmentation_output_range():
    """Augmented image values should stay in valid range."""
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
    ])
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    for _ in range(50):
        result = transform(image=image)
        assert result["image"].min() >= 0
        assert result["image"].max() <= 255
```

### Testing Video Processing

```python
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """Create a synthetic video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    for i in range(90):  # 3 seconds at 30fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a moving circle for visual verification
        x = int(320 + 200 * np.sin(2 * np.pi * i / 90))
        cv2.circle(frame, (x, 240), 30, (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return video_path


def test_video_reader_frame_count(synthetic_video):
    """VideoReader should report correct frame count."""
    reader = VideoReader(synthetic_video)
    assert reader.frame_count == 90
    assert reader.fps == pytest.approx(30.0, abs=0.1)
    assert reader.resolution == (640, 480)


def test_video_reader_iteration(synthetic_video):
    """VideoReader should yield all frames."""
    reader = VideoReader(synthetic_video)
    frames = list(reader)
    assert len(frames) == 90
    assert all(f.shape == (480, 640, 3) for f in frames)
```

## Mocking External Dependencies

Use mocking to test code that depends on external services, GPUs, or expensive resources.

```python
from unittest.mock import MagicMock, patch

import pytest


def test_wandb_logging_called():
    """Verify metrics are logged to W&B."""
    with patch("myproject.training.wandb") as mock_wandb:
        trainer = Trainer(use_wandb=True)
        trainer.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)

        mock_wandb.log.assert_called_once_with(
            {"loss": 0.5, "accuracy": 0.9}, step=100
        )


def test_model_loading_from_checkpoint(tmp_path):
    """Test model loading without requiring actual checkpoint file."""
    mock_checkpoint = {
        "model_state_dict": {k: torch.randn_like(v) for k, v in model.state_dict().items()},
        "epoch": 50,
        "loss": 0.1,
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(mock_checkpoint, checkpoint_path)

    loaded_model = MyModel.load_from_checkpoint(checkpoint_path)
    assert loaded_model is not None


@pytest.fixture
def mock_camera():
    """Mock camera that returns synthetic frames."""
    camera = MagicMock()
    camera.read.return_value = (
        True,
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    )
    camera.isOpened.return_value = True
    camera.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30,
    }.get(prop, 0)
    return camera


def test_camera_processor_with_mock(mock_camera):
    """Test camera processing pipeline with mock camera."""
    with patch("cv2.VideoCapture", return_value=mock_camera):
        processor = CameraProcessor(camera_id=0)
        frame = processor.read_frame()
        assert frame.shape == (480, 640, 3)
```

## Performance Tests

Mark performance tests separately so they can be skipped in quick test runs.

```python
import time

import pytest


@pytest.mark.slow
def test_model_inference_speed(device):
    """Model inference should be under 50ms per image."""
    model = MyModel(num_classes=80).to(device).eval()
    x = torch.randn(1, 3, 640, 640, device=device)

    # Warmup
    for _ in range(10):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / 100) * 1000
    assert ms_per_image < 50, f"Inference too slow: {ms_per_image:.1f}ms"


@pytest.mark.slow
def test_data_loading_throughput(tmp_data_dir):
    """DataLoader should achieve at least 100 images/second."""
    dataset = ImageDataset(tmp_data_dir / "train")
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    start = time.perf_counter()
    total_images = 0
    for batch in loader:
        total_images += batch.shape[0]
    elapsed = time.perf_counter() - start

    throughput = total_images / elapsed
    assert throughput > 100, f"Too slow: {throughput:.0f} img/s"
```

## Edge Case Testing

```python
def test_empty_image():
    """Model should handle zero-size image gracefully."""
    model = MyModel(num_classes=10)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        model(torch.randn(1, 3, 0, 0))


def test_single_pixel_image():
    """Model should handle 1x1 images."""
    model = MyModel(num_classes=10)
    output = model(torch.randn(1, 3, 1, 1))
    assert output.shape == (1, 10)


def test_no_detections():
    """Post-processing should return empty results when nothing detected."""
    raw_output = torch.zeros(1, 0, 6)  # No detections
    results = postprocess(raw_output, confidence_threshold=0.5)
    assert len(results[0]["boxes"]) == 0
    assert len(results[0]["labels"]) == 0


def test_all_same_class():
    """Metrics should handle case where all predictions are same class."""
    preds = np.array([0, 0, 0, 0, 0])
    targets = np.array([0, 1, 2, 0, 1])
    metrics = compute_metrics(preds, targets, num_classes=3)
    assert 0 <= metrics["accuracy"] <= 1


def test_nan_in_loss():
    """Training should detect and handle NaN loss."""
    with pytest.raises(RuntimeError, match="NaN"):
        trainer = Trainer(detect_nan=True)
        trainer.train_step(
            model, torch.tensor(float("nan")), optimizer
        )
```

## Coverage Configuration

Configure coverage in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
]
addopts = [
    "--strict-markers",
    "-ra",
    "--tb=short",
]

[tool.coverage.run]
source = ["src/myproject"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

Run with:

```bash
# Run all tests with coverage
pytest --cov=src/myproject --cov-report=html --cov-report=term-missing

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v --tb=long

# Run specific test file
pytest tests/unit/test_model.py

# Run tests matching a pattern
pytest -k "test_bbox"
```

## CI Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov=src/myproject --cov-report=xml -m "not slow"
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

## Best Practices

1. **Test behavior, not implementation** -- Tests should verify what a function does, not how it does it. This makes refactoring safe.
2. **One assertion concept per test** -- Each test should verify one logical concept. Multiple related `assert` statements are fine if they test the same thing.
3. **Use descriptive test names** -- `test_model_raises_on_wrong_input_channels` is better than `test_model_error`.
4. **Keep tests fast** -- Unit tests should run in milliseconds. Mark slow tests with `@pytest.mark.slow`.
5. **Use fixtures over setup methods** -- Pytest fixtures are more flexible and composable than `setUp`/`tearDown`.
6. **Test edge cases** -- Empty inputs, single elements, maximum sizes, NaN values, wrong dtypes.
7. **Pin random seeds in tests** -- Use a seed fixture to make tests deterministic.
8. **Maintain 80%+ coverage** -- Configure `fail_under = 80` in coverage settings.
9. **Run tests in CI on every push** -- Never merge without green tests.
10. **Test data shapes explicitly** -- In ML code, shape mismatches are the most common bugs. Assert shapes everywhere.
