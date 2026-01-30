# Testing Skill

## Purpose

This skill provides comprehensive pytest patterns tailored for ML and computer vision projects. It covers test organization, fixtures, parametrized tests, CV-specific testing strategies (synthetic data, augmentation validation, video processing), mocking, performance benchmarks, and CI integration.

## When to Use

- You need to write tests for model forward passes, data pipelines, or augmentation logic
- You are setting up test infrastructure for a new ML/CV project
- You want to test code that depends on cameras, GPUs, or external services
- You need to validate bounding box conversions, image transformations, or metric calculations
- You are configuring pytest, coverage, and CI for an ML codebase

## Key Patterns

- **AAA pattern**: Arrange-Act-Assert for readable tests
- **Synthetic data fixtures**: Generate test images, bounding boxes, and video without real datasets
- **Parametrized tests**: Test multiple input sizes, formats, and model architectures
- **Mocking**: Replace cameras, experiment trackers, and GPU dependencies
- **Performance marks**: Separate slow benchmarks from fast unit tests with `@pytest.mark.slow`
- **Shape assertions**: Explicitly verify tensor and array shapes throughout

## Usage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/myproject --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Run specific test pattern
pytest -k "test_bbox"
```

## Benefits

- Catches silent ML failures (wrong shapes, broken augmentations, label mismatches) before they waste GPU time
- Synthetic data fixtures make tests fast and self-contained with no external data dependencies
- Parametrized tests cover many input combinations without code duplication
- Coverage enforcement ensures critical code paths are tested

## See Also

- `SKILL.md` in this directory for full documentation and code examples
- `github-actions` skill for CI workflow configuration
