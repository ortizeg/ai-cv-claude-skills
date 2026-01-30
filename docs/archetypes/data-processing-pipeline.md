# Data Processing Pipeline

ETL workflows for dataset preparation, transformation, and validation with parallel processing and DVC integration.

## Purpose

This archetype structures dataset processing as a series of well-defined stages: download, preprocess, validate, split, and package. Each stage is a self-contained module with Pydantic-validated configuration, making pipelines reproducible and composable. DVC tracks large data assets, and parallel processing handles large-scale datasets efficiently.

## Directory Structure

```
{{project_slug}}/
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract stage interface
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── validate.py
│   │   ├── split.py
│   │   └── export.py
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── image.py
│   └── schemas/
│       ├── __init__.py
│       └── dataset.py         # Pydantic data schemas
├── configs/
│   ├── pipeline.yaml
│   └── stages/
│       ├── download.yaml
│       └── preprocess.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── dvc.yaml                   # DVC pipeline definition
├── tests/
│   ├── test_stages.py
│   └── test_pipeline.py
└── ...
```

## Pipeline Stage Interface

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class StageConfig(BaseModel):
    """Base configuration for pipeline stages."""
    input_dir: str
    output_dir: str

class Stage(ABC):
    @abstractmethod
    def run(self, config: StageConfig) -> None: ...
    @abstractmethod
    def validate(self) -> bool: ...
```

## Usage

```bash
# Run full pipeline
pixi run python -m my_project.pipeline

# Run single stage
pixi run python -m my_project.pipeline stage=preprocess

# Reproduce with DVC
dvc repro
```

## Customization

- Add new stages in `src/{{package_name}}/stages/`
- Define stage configs in `configs/stages/`
- Add image transforms in `transforms/`
- Configure DVC remotes for data storage
