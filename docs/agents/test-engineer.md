# Test Engineer Agent

Automated testing and coverage enforcement that runs as a GitHub Action. Must pass before any PR can be merged.

## Strictness Level

**Blocking** -- all tests must pass and coverage thresholds must be met.

## Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| Core logic (models, pipelines) | 90% |
| Data pipelines | 80% |
| Utilities | 70% |
| **Overall project** | **80%** |

## Test Structure

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
├── integration/           # Multi-component tests
│   ├── test_training.py
│   └── test_pipeline.py
├── e2e/                   # End-to-end tests
│   └── test_full_workflow.py
├── conftest.py            # Shared fixtures
└── test_data/             # Test fixtures (sample images, configs)
```

## Test Patterns

### AAA Pattern (Arrange, Act, Assert)

```python
def test_model_forward_pass():
    # Arrange
    config = ModelConfig(input_channels=3, num_classes=10)
    model = MyModel(config)
    batch = torch.randn(2, 3, 32, 32)

    # Act
    output = model(batch)

    # Assert
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
```

### Parametrized Tests

```python
@pytest.mark.parametrize("batch_size,height,width", [
    (1, 32, 32), (4, 64, 64), (8, 224, 224),
])
def test_model_input_sizes(batch_size, height, width):
    model = MyModel(config)
    output = model(torch.randn(batch_size, 3, height, width))
    assert output.shape[0] == batch_size
```

### CV-Specific: Synthetic Data

```python
def test_detector_with_synthetic_image():
    image = torch.rand(3, 640, 640)
    detector = YOLODetector(config)
    detections = detector.detect(image)

    for det in detections:
        assert 0 <= det.bbox.confidence <= 1
        assert det.bbox.width > 0
```

## GitHub Action Setup

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1
        with:
          pixi-version: latest
      - run: uv run pytest --cov=src --cov-fail-under=80
```

## Local Usage

```bash
# Run all tests
uv run test

# Run with coverage report
uv run test-cov

# Run specific test file
uv run pytest tests/unit/test_models.py -v

# Run tests matching pattern
uv run pytest -k "test_model"
```

## Additional Requirements

- **No skipped tests** -- fix or remove them
- **Tests must be typed** -- type hints on test functions
- **Fast tests** -- unit tests should run in milliseconds
- **Independent tests** -- no test should depend on another
