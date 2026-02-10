# Testing

The Testing skill defines a comprehensive test strategy for ML/CV codebases, covering unit tests, integration tests, and ML-specific validation patterns using pytest.

**Skill directory:** `skills/testing/`

## Purpose

ML code is notoriously undertested. Models "work" until they silently produce garbage due to a transposed tensor or a misconfigured augmentation. This skill teaches Claude Code to write tests that catch these issues: shape checks, determinism tests, loss convergence smoke tests, and data pipeline validation. It also covers pytest fixtures for GPU-aware testing and synthetic dataset generation.

## When to Use

- Every project. Testing is not optional, even for research code.
- Particularly important for training pipelines, data loaders, and inference services.

## Key Patterns

### Test Structure

```
tests/
    conftest.py              # Shared fixtures
    test_model.py            # Model unit tests
    test_data.py             # Data pipeline tests
    test_transforms.py       # Augmentation tests
    test_integration.py      # End-to-end training step tests
    test_inference.py        # Inference pipeline tests
```

### Model Shape Tests

```python
import pytest
import torch

from my_project.model import ImageClassifier

@pytest.fixture
def model() -> ImageClassifier:
    return ImageClassifier(num_classes=10, backbone="resnet18")

def test_forward_output_shape(model: ImageClassifier) -> None:
    batch = torch.randn(4, 3, 224, 224)
    output = model(batch)
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"

def test_forward_no_nan(model: ImageClassifier) -> None:
    batch = torch.randn(4, 3, 224, 224)
    output = model(batch)
    assert not torch.isnan(output).any(), "Model output contains NaN"
```

### Training Step Smoke Test

```python
def test_single_training_step(model: ImageClassifier) -> None:
    """Verify a single training step runs without error and produces a scalar loss."""
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "label": torch.randint(0, 10, (2,)),
    }
    loss = model.training_step(batch, batch_idx=0)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    loss.backward()  # Verify gradients flow
```

### Data Pipeline Tests

```python
def test_datamodule_batch_shapes(data_module: ImageDataModule) -> None:
    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))
    assert batch["image"].shape[1:] == (3, 224, 224)
    assert batch["label"].dtype == torch.long
```

### Pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests requiring GPU",
]
filterwarnings = ["ignore::DeprecationWarning"]
```

## Anti-Patterns to Avoid

- Do not skip ML tests because "the model is too slow" -- use small models and synthetic data
- Do not test only happy paths -- test edge cases like empty batches, single samples, and mismatched dimensions
- Do not use `assert True` after calling a function -- check actual outputs
- Do not share mutable state between tests -- use fixtures with proper scope

## Combines Well With

- **PyTorch Lightning** -- Test LightningModule hooks and DataModule setup
- **Code Quality** -- Type-checked test code
- **GitHub Actions** -- Run tests in CI on every PR
- **Pre-commit** -- Run fast tests before commit

## Full Reference

See [`skills/testing/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/testing/SKILL.md) for advanced patterns including GPU-conditional tests, snapshot testing for model outputs, and property-based testing with Hypothesis.
