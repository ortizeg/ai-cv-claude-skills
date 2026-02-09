---
name: aws-sagemaker
description: >
  AWS SageMaker patterns for ML training and deployment. Covers training jobs,
  real-time endpoints, batch transform, processing jobs, model registry,
  hyperparameter tuning, SageMaker Pipelines, and S3 data management.
---

# AWS SageMaker Skill

You are building ML training and deployment pipelines on AWS SageMaker. Follow these patterns exactly.

## Core Philosophy

SageMaker provides managed infrastructure for training, tuning, and deploying ML models at scale. Every cloud training and deployment workflow targeting AWS uses SageMaker. Use the SageMaker Python SDK for all interactions — never call low-level boto3 APIs for SageMaker operations unless the SDK does not support a feature.

## Project Structure

### Standard SageMaker Project Layout

```
project/
├── src/
│   ├── training/
│   │   ├── train.py              # Training entry point
│   │   ├── model.py              # Model definition
│   │   └── data.py               # Data loading
│   ├── inference/
│   │   ├── inference.py          # model_fn, input_fn, predict_fn, output_fn
│   │   └── requirements.txt      # Inference dependencies
│   └── processing/
│       └── preprocess.py         # Processing job script
├── pipelines/
│   ├── training_pipeline.py      # SageMaker Pipeline definition
│   └── config.py                 # Pipeline configuration
├── configs/
│   ├── training.yaml             # Hyperparameters
│   └── infrastructure.yaml       # Instance types, counts
└── tests/
    ├── test_training_local.py    # Local mode tests
    └── test_inference.py         # Endpoint tests
```

## Training Jobs

### Configuring a Training Job with PyTorch Estimator

```python
"""SageMaker training job configuration."""

from __future__ import annotations

from pathlib import Path

import sagemaker
from pydantic import BaseModel, Field
from sagemaker.pytorch import PyTorch


class TrainingConfig(BaseModel, frozen=True):
    """Training job configuration."""

    role: str
    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    max_run_seconds: int = 86400
    volume_size_gb: int = 100
    output_s3_uri: str
    base_job_name: str = "cv-training"
    framework_version: str = "2.1.0"
    py_version: str = "py310"


class HyperParameters(BaseModel, frozen=True):
    """Training hyperparameters passed to the training script."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_name: str = "resnet50"
    num_classes: int = 10


def create_estimator(
    config: TrainingConfig,
    hyperparameters: HyperParameters,
) -> PyTorch:
    """Create a SageMaker PyTorch estimator."""
    return PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role=config.role,
        instance_type=config.instance_type,
        instance_count=config.instance_count,
        framework_version=config.framework_version,
        py_version=config.py_version,
        output_path=config.output_s3_uri,
        base_job_name=config.base_job_name,
        max_run=config.max_run_seconds,
        volume_size=config.volume_size_gb,
        hyperparameters=hyperparameters.model_dump(),
        environment={
            "NCCL_DEBUG": "INFO",
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        },
        distribution={
            "torch_distributed": {"enabled": True}
        } if config.instance_count > 1 else None,
    )


def launch_training(
    config: TrainingConfig,
    hyperparameters: HyperParameters,
    train_s3_uri: str,
    val_s3_uri: str,
) -> str:
    """Launch a SageMaker training job and return the job name."""
    estimator = create_estimator(config, hyperparameters)

    estimator.fit(
        inputs={
            "train": train_s3_uri,
            "validation": val_s3_uri,
        },
        wait=False,
    )

    return estimator.latest_training_job.name
```

### Training Script Entry Point

```python
"""SageMaker training script entry point (src/training/train.py)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger


def parse_args() -> argparse.Namespace:
    """Parse SageMaker-injected arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--num-classes", type=int, default=10)

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    logger.info("Starting training with args: {}", vars(args))

    # Distributed setup
    world_size = int(os.environ.get("SM_NUM_GPUS", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info("Distributed training: rank {} of {}", local_rank, world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model_name, args.num_classes).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ... training loop ...

    # Save model artifacts
    model_path = Path(args.model_dir) / "model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to {}", model_path)

    # Save metrics
    metrics = {"final_val_loss": 0.25, "final_val_acc": 0.92}
    metrics_path = Path(args.output_data_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

## Real-Time Endpoints

### Deploying a Model Endpoint

```python
"""SageMaker endpoint deployment."""

from __future__ import annotations

from pydantic import BaseModel, Field
from sagemaker.pytorch import PyTorchModel


class EndpointConfig(BaseModel, frozen=True):
    """Endpoint deployment configuration."""

    role: str
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    endpoint_name: str
    model_data_s3: str
    framework_version: str = "2.1.0"
    py_version: str = "py310"


def deploy_endpoint(config: EndpointConfig) -> str:
    """Deploy a real-time inference endpoint."""
    model = PyTorchModel(
        model_data=config.model_data_s3,
        role=config.role,
        framework_version=config.framework_version,
        py_version=config.py_version,
        entry_point="inference.py",
        source_dir="src/inference",
    )

    predictor = model.deploy(
        initial_instance_count=config.instance_count,
        instance_type=config.instance_type,
        endpoint_name=config.endpoint_name,
    )

    return predictor.endpoint_name
```

### Custom Inference Script

```python
"""Custom inference handlers (src/inference/inference.py).

SageMaker calls these functions in order:
  input_fn → predict_fn → output_fn
"""

from __future__ import annotations

import io
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def model_fn(model_dir: str) -> torch.nn.Module:
    """Load the trained model from the model directory."""
    model = build_model("resnet50", num_classes=10)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    """Deserialize input data to a tensor."""
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    elif content_type == "application/json":
        data = json.loads(request_body)
        return torch.tensor(data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: torch.Tensor, model: torch.nn.Module) -> dict:
    """Run inference on the input tensor."""
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    with torch.no_grad():
        outputs = model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return {
        "predictions": predictions.cpu().numpy().tolist(),
        "probabilities": probabilities.cpu().numpy().tolist(),
    }


def output_fn(prediction: dict, accept: str) -> str:
    """Serialize prediction output."""
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
```

## Hyperparameter Tuning

### Automatic Model Tuning

```python
"""SageMaker hyperparameter tuning job."""

from __future__ import annotations

from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)


def create_tuner(estimator: PyTorch) -> HyperparameterTuner:
    """Create a hyperparameter tuning job."""
    hyperparameter_ranges = {
        "learning-rate": ContinuousParameter(1e-5, 1e-2, scaling_type="Logarithmic"),
        "batch-size": CategoricalParameter([16, 32, 64, 128]),
        "weight-decay": ContinuousParameter(1e-6, 1e-2, scaling_type="Logarithmic"),
        "epochs": IntegerParameter(10, 100),
    }

    return HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:accuracy",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[
            {"Name": "validation:accuracy", "Regex": r"val_acc=(\S+)"},
            {"Name": "validation:loss", "Regex": r"val_loss=(\S+)"},
        ],
        max_jobs=20,
        max_parallel_jobs=4,
        strategy="Bayesian",
        objective_type="Maximize",
    )
```

## SageMaker Pipelines

### End-to-End Training Pipeline

```python
"""SageMaker Pipeline for training and registration."""

from __future__ import annotations

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def create_pipeline(
    role: str,
    pipeline_name: str = "cv-training-pipeline",
) -> Pipeline:
    """Create a SageMaker Pipeline."""
    session = sagemaker.Session()

    # Pipeline parameters
    input_data = ParameterString(name="InputData")
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.90)

    # Step 1: Data preprocessing
    processor = ScriptProcessor(
        role=role,
        image_uri=session.sagemaker_client.describe_image("pytorch-training")["ImageUri"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        command=["python3"],
    )

    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="val", source="/opt/ml/processing/output/val"),
        ],
        code="src/processing/preprocess.py",
    )

    # Step 2: Training
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role=role,
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
    )

    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            "validation": preprocess_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
        },
    )

    # Step 3: Conditional model registration
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=training_step.name,
            property_file="metrics",
            json_path="val_accuracy",
        ),
        right=accuracy_threshold,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=estimator.register(
            content_types=["application/json", "application/x-image"],
            response_types=["application/json"],
            model_package_group_name="cv-models",
            approval_status="PendingManualApproval",
        ),
    )

    condition_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[input_data, accuracy_threshold],
        steps=[preprocess_step, training_step, condition_step],
        sagemaker_session=session,
    )
```

## S3 Data Management

### Data Upload and Download Patterns

```python
"""S3 data management for SageMaker workflows."""

from __future__ import annotations

from pathlib import Path

import boto3
import sagemaker
from loguru import logger


def upload_dataset(
    local_path: Path,
    bucket: str,
    prefix: str = "datasets",
) -> str:
    """Upload a local dataset to S3 and return the S3 URI."""
    session = sagemaker.Session()
    s3_uri = session.upload_data(
        path=str(local_path),
        bucket=bucket,
        key_prefix=prefix,
    )
    logger.info("Uploaded {} to {}", local_path, s3_uri)
    return s3_uri


def download_model_artifacts(
    model_data_s3: str,
    local_dir: Path,
) -> Path:
    """Download model artifacts from S3."""
    local_dir.mkdir(parents=True, exist_ok=True)

    session = boto3.Session()
    s3 = session.resource("s3")

    # Parse S3 URI
    parts = model_data_s3.replace("s3://", "").split("/", 1)
    bucket_name, key = parts[0], parts[1]

    local_path = local_dir / Path(key).name
    s3.Bucket(bucket_name).download_file(key, str(local_path))

    logger.info("Downloaded {} to {}", model_data_s3, local_path)
    return local_path
```

## Batch Transform

### Offline Batch Inference

```python
"""SageMaker batch transform for offline inference."""

from __future__ import annotations

from sagemaker.pytorch import PyTorchModel


def run_batch_transform(
    model_data_s3: str,
    input_s3_uri: str,
    output_s3_uri: str,
    role: str,
    instance_type: str = "ml.g5.xlarge",
) -> None:
    """Run batch transform on a dataset."""
    model = PyTorchModel(
        model_data=model_data_s3,
        role=role,
        framework_version="2.1.0",
        py_version="py310",
        entry_point="inference.py",
        source_dir="src/inference",
    )

    transformer = model.transformer(
        instance_count=1,
        instance_type=instance_type,
        output_path=output_s3_uri,
        strategy="MultiRecord",
        max_payload=6,
    )

    transformer.transform(
        data=input_s3_uri,
        content_type="application/json",
        split_type="Line",
    )
```

## Local Mode Testing

### Test Training Locally Before Submitting to SageMaker

```python
"""Local mode testing for SageMaker training scripts."""

from __future__ import annotations

import pytest
from sagemaker.pytorch import PyTorch


@pytest.fixture
def local_estimator() -> PyTorch:
    """Create a local mode estimator for testing."""
    return PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role="arn:aws:iam::000000000000:role/dummy",
        instance_type="local",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters={
            "epochs": 1,
            "batch-size": 4,
            "model-name": "resnet18",
        },
    )


def test_training_local(local_estimator: PyTorch, tmp_path: Path) -> None:
    """Verify training script runs in local mode."""
    # Create minimal test data
    create_test_dataset(tmp_path / "train")
    create_test_dataset(tmp_path / "val")

    local_estimator.fit({
        "train": f"file://{tmp_path / 'train'}",
        "validation": f"file://{tmp_path / 'val'}",
    })
```

## Anti-Patterns

- **Never hardcode S3 paths** — use SageMaker session defaults and `ParameterString` in pipelines.
- **Never use `instance_type="ml.p3.16xlarge"` for simple models** — right-size instances. Start with `ml.g5.xlarge` and scale up.
- **Never skip local mode testing** — always test with `instance_type="local"` before submitting cloud jobs.
- **Never put credentials in training scripts** — SageMaker injects the IAM role automatically.
- **Never download the full dataset inside the training script** — use SageMaker input channels (`SM_CHANNEL_*`).
- **Never forget to call `wait=False` for long training jobs** — use async job submission and poll status separately.

## Integration with Other Skills

- **PyTorch Lightning** — Training scripts use LightningModule inside SageMaker training jobs.
- **Hydra Config** — Hyperparameters serialized and passed to SageMaker as flat key-value pairs.
- **W&B / MLflow** — Experiment tracking inside SageMaker training containers.
- **Docker CV** — Custom training containers when the built-in SageMaker images are insufficient.
- **DVC** — Data versioning with S3 remote storage that SageMaker can access.
