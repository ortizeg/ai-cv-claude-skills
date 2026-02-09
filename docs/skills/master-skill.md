# Master Skill

The Master Skill defines the universal coding conventions that apply to every CV/ML project in this framework. It is the foundational skill that all other skills build upon.

**Skill directory:** `skills/master-skill/`

## Purpose

The Master Skill establishes baseline standards for Python code in computer vision and deep learning projects. It covers import ordering, naming conventions, type annotation requirements, docstring format, error handling patterns, and project layout. Loading this skill ensures Claude Code produces consistent, professional code regardless of which other skills are active.

## When to Use

Always. The Master Skill should be loaded in every session. Other skills assume its conventions are active.

## Key Conventions

### Import Ordering

```python
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from my_project.config import ModelConfig
from my_project.utils import setup_logging
```

Imports follow the pattern: `__future__` annotations, standard library, third-party, local -- each group separated by a blank line.

### Type Annotations

All function signatures require type annotations. Use `from __future__ import annotations` at the top of every module for modern annotation syntax:

```python
def process_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    normalize: bool = True,
) -> np.ndarray:
    """Process a single image for model input."""
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `data_loader.py` |
| Classes | `PascalCase` | `ImageClassifier` |
| Functions | `snake_case` | `load_checkpoint` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_IMAGE_SIZE` |
| Type aliases | `PascalCase` | `BatchType = dict[str, Tensor]` |

### Logging

Use structured logging instead of print statements:

```python
logger = logging.getLogger(__name__)

logger.info("Training started", extra={"epoch": epoch, "lr": lr})
```

### Error Handling

Raise specific exceptions with informative messages:

```python
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

if image.ndim != 3:
    raise ValueError(f"Expected 3D image array, got shape {image.shape}")
```

## Integration with Other Skills

The Master Skill's conventions are assumed by every other skill. When the PyTorch Lightning skill specifies a `LightningModule` template, it uses the Master Skill's import ordering and type annotation style. When the Testing skill defines test structure, it follows the Master Skill's naming conventions.

## Full Reference

See the complete skill definition in [`skills/master-skill/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/master-skill/SKILL.md) for the full set of conventions, including docstring format, project layout rules, and anti-patterns to avoid.
