# Test Engineer Agent

Automated testing and coverage enforcement.

## Strictness Level

**BLOCKING** — Must pass to merge

## Requirements

- All tests must pass
- Coverage >80% overall
- No skipped tests
- Tests must be typed

## What It Tests

1. **Unit Tests** — Individual components
2. **Integration Tests** — Multi-component
3. **Coverage** — Code coverage metrics
4. **Performance** — Speed benchmarks (if present)

## Local Usage

```bash
pixi run test
pixi run test-cov
pixi run pytest tests/unit/test_models.py
```
