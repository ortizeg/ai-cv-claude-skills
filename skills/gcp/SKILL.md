# GCP Skill

Google Cloud Platform services for CV/ML projects: Artifact Registry, Cloud Storage, Vertex AI training, and Docker image management.

## Artifact Registry

Use Artifact Registry for Docker images and Python packages. It replaces the deprecated Container Registry.

### Create a Docker Repository

```bash
# Create a Docker repository in Artifact Registry
gcloud artifacts repositories create ml-images \
    --repository-format=docker \
    --location=us-central1 \
    --description="CV/ML training and inference images"

# Verify creation
gcloud artifacts repositories list --location=us-central1
```

### Push and Pull Docker Images

```bash
# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Tag and push
docker tag my-training:latest \
    us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0

docker push us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0

# Pull
docker pull us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0
```

### Python Package Repository

```bash
# Create a Python repository
gcloud artifacts repositories create ml-packages \
    --repository-format=python \
    --location=us-central1

# Configure pip to pull from Artifact Registry
gcloud artifacts print-settings python \
    --repository=ml-packages \
    --location=us-central1
```

```python
# pixi.toml — add Artifact Registry as extra index
# [project]
# name = "my-cv-project"
#
# [tool.pixi.pypi-options]
# extra-index-urls = [
#     "https://us-central1-python.pkg.dev/my-project/ml-packages/simple/"
# ]
```

## Cloud Storage

Use Cloud Storage for datasets, model checkpoints, and training artifacts.

### Upload and Download with gsutil

```bash
# Upload a dataset
gsutil -m cp -r ./data/coco/ gs://my-ml-bucket/datasets/coco/

# Download model checkpoint
gsutil cp gs://my-ml-bucket/checkpoints/resnet50_epoch20.pt ./checkpoints/

# Sync outputs (only transfers changed files)
gsutil -m rsync -r ./outputs/ gs://my-ml-bucket/runs/experiment-42/
```

### Python Client

```python
from google.cloud import storage


def upload_model_artifact(
    bucket_name: str,
    source_path: str,
    destination_blob: str,
) -> str:
    """Upload a model artifact to Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_path)
    return f"gs://{bucket_name}/{destination_blob}"


def download_checkpoint(
    bucket_name: str,
    blob_name: str,
    destination_path: str,
) -> None:
    """Download a checkpoint from Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path)
```

### Mount with gcsfuse

```bash
# Mount a bucket as a local directory (useful for large datasets)
gcsfuse --implicit-dirs my-ml-bucket /mnt/gcs-data

# Use in a training container
docker run --privileged \
    -v /mnt/gcs-data:/data:ro \
    my-training:latest
```

```dockerfile
# Install gcsfuse in a training container
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-jammy main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add - && \
    apt-get update && \
    apt-get install -y gcsfuse && \
    rm -rf /var/lib/apt/lists/*
```

## Vertex AI Training Jobs

Submit custom training jobs to Vertex AI for GPU-accelerated model training.

### Custom Training Job

```python
from google.cloud import aiplatform


def submit_training_job(
    project: str,
    location: str,
    display_name: str,
    container_uri: str,
    args: list[str],
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    staging_bucket: str = "",
) -> aiplatform.CustomJob:
    """Submit a custom training job to Vertex AI."""
    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path="src/train.py",
        container_uri=container_uri,
        args=args,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )
    job.run(sync=False)
    return job
```

### Custom Container Training

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="resnet50-finetune",
    container_uri="us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0",
    command=["python", "-m", "my_project.train"],
    model_serving_container_image_uri=(
        "us-central1-docker.pkg.dev/my-project/ml-images/inference:v1.2.0"
    ),
)

model = job.run(
    args=["--config=configs/finetune.yaml", "--epochs=50"],
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=2,
    base_output_dir="gs://my-ml-bucket/vertex-outputs/",
)
```

### GPU Selection Reference

| GPU Type | Vertex AI Name | Use Case |
|----------|----------------|----------|
| T4 | `NVIDIA_TESLA_T4` | Inference, small-batch training |
| V100 | `NVIDIA_TESLA_V100` | Training (16 GB HBM2) |
| A100 40GB | `NVIDIA_TESLA_A100` | Large-scale training |
| A100 80GB | `NVIDIA_A100_80GB` | Large models, large batch sizes |
| L4 | `NVIDIA_L4` | Inference, cost-effective training |
| H100 | `NVIDIA_H100_80GB` | Highest throughput training |

### Prebuilt Training Containers

```python
# Use Google's prebuilt PyTorch containers instead of custom images
PYTORCH_GPU_CONTAINER = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest"

job = aiplatform.CustomJob.from_local_script(
    display_name="detection-train",
    script_path="src/train.py",
    container_uri=PYTORCH_GPU_CONTAINER,
    requirements=["torchvision", "albumentations", "pycocotools"],
    args=["--epochs=100", "--batch-size=32"],
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

## Docker Image Management

Build, tag, and push images to Artifact Registry for Vertex AI and GKE workloads.

### Build and Push Workflow

```bash
# Variables
PROJECT_ID="my-project"
REGION="us-central1"
REPO="ml-images"
IMAGE="training"
TAG="v1.2.0"
FULL_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

# Build with cache
docker build \
    --tag "${FULL_URI}" \
    --cache-from "${FULL_URI}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -f Dockerfile.training .

# Push
docker push "${FULL_URI}"
```

### Multi-Stage for Training and Inference

```dockerfile
# ==============================================================================
# Base stage — shared between training and inference
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Training stage — full environment with dev tools
# ==============================================================================
FROM base AS training

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --frozen

COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/

RUN pixi run pip install -e ".[dev]"
RUN useradd -m -u 1000 trainer
USER trainer

ENTRYPOINT ["pixi", "run", "python", "-m"]
CMD ["my_project.train"]

# ==============================================================================
# Inference stage — minimal runtime
# ==============================================================================
FROM base AS inference

WORKDIR /app
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/ src/

RUN useradd -m -u 1000 appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GitHub Actions: Build and Push to Artifact Registry

```yaml
# .github/workflows/docker-gcp.yml
name: Build and Push to Artifact Registry

on:
  push:
    tags: ["v*"]

env:
  PROJECT_ID: my-project
  REGION: us-central1
  REPOSITORY: ml-images

jobs:
  build-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}

      - uses: google-github-actions/setup-gcloud@v2

      - run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/training:${{ github.ref_name }}
            ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/training:latest
          target: training
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Authentication

### Service Account Setup

```bash
# Create a service account for training jobs
gcloud iam service-accounts create ml-trainer \
    --display-name="ML Training Service Account"

# Grant required roles
SA_EMAIL="ml-trainer@my-project.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.reader"
```

### Workload Identity Federation for CI

```bash
# Create a workload identity pool for GitHub Actions
gcloud iam workload-identity-pools create "github-pool" \
    --location="global" \
    --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
    --location="global" \
    --workload-identity-pool="github-pool" \
    --display-name="GitHub Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --issuer-uri="https://token.actions.githubusercontent.com"

# Allow the GitHub repo to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/OWNER/REPO"
```

### Docker Authentication

```bash
# ✅ Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# ✅ For CI: use credential helper with service account key
# (prefer Workload Identity Federation when possible)
cat key.json | docker login -u _json_key --password-stdin \
    https://us-central1-docker.pkg.dev

# ❌ Do not use gcloud auth print-access-token for long-running processes
# Tokens expire after 1 hour
```

## Pydantic Configuration

Define typed configuration models for GCP project settings and job specifications.

```python
from pydantic import BaseModel, Field


class GCPConfig(BaseModel, frozen=True):
    """GCP project configuration."""

    project_id: str = Field(description="GCP project ID")
    region: str = Field(default="us-central1", description="Default region")
    zone: str = Field(default="us-central1-a", description="Default zone")


class ArtifactRegistryConfig(BaseModel, frozen=True):
    """Artifact Registry repository settings."""

    repository: str = Field(description="Repository name")
    location: str = Field(default="us-central1")

    def docker_uri(self, project_id: str, image: str, tag: str) -> str:
        """Build the full Docker image URI."""
        return (
            f"{self.location}-docker.pkg.dev/{project_id}"
            f"/{self.repository}/{image}:{tag}"
        )


class StorageBucketConfig(BaseModel, frozen=True):
    """Cloud Storage bucket configuration."""

    bucket_name: str = Field(description="GCS bucket name")
    datasets_prefix: str = Field(default="datasets/")
    checkpoints_prefix: str = Field(default="checkpoints/")
    outputs_prefix: str = Field(default="outputs/")

    def dataset_uri(self, name: str) -> str:
        return f"gs://{self.bucket_name}/{self.datasets_prefix}{name}"

    def checkpoint_uri(self, name: str) -> str:
        return f"gs://{self.bucket_name}/{self.checkpoints_prefix}{name}"


class VertexJobConfig(BaseModel, frozen=True):
    """Vertex AI training job configuration."""

    display_name: str
    machine_type: str = Field(default="n1-standard-8")
    accelerator_type: str = Field(default="NVIDIA_TESLA_T4")
    accelerator_count: int = Field(default=1, ge=1, le=8)
    replica_count: int = Field(default=1, ge=1)
    staging_bucket: str = Field(description="GCS bucket for staging")
    boot_disk_size_gb: int = Field(default=100, ge=50)


class GCPProjectConfig(BaseModel, frozen=True):
    """Complete GCP configuration for an ML project."""

    gcp: GCPConfig
    artifact_registry: ArtifactRegistryConfig
    storage: StorageBucketConfig
    vertex_job: VertexJobConfig
```

```yaml
# configs/gcp.yaml — Hydra-compatible configuration
gcp:
  project_id: my-cv-project
  region: us-central1
  zone: us-central1-a

artifact_registry:
  repository: ml-images
  location: us-central1

storage:
  bucket_name: my-cv-project-ml
  datasets_prefix: datasets/
  checkpoints_prefix: checkpoints/
  outputs_prefix: outputs/

vertex_job:
  display_name: resnet50-train
  machine_type: n1-standard-8
  accelerator_type: NVIDIA_TESLA_T4
  accelerator_count: 1
  staging_bucket: my-cv-project-ml-staging
  boot_disk_size_gb: 100
```

## Integration with pixi

Define pixi tasks for common GCP operations to ensure consistency across the team.

```toml
# pixi.toml — GCP task definitions
[project]
name = "my-cv-project"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
google-cloud-storage = ">=2.14"
google-cloud-aiplatform = ">=1.40"

[feature.dev.dependencies]
google-cloud-artifact-registry = ">=1.11"

[tasks]
# Authentication
gcp-auth = "gcloud auth application-default login"
gcp-docker-auth = "gcloud auth configure-docker us-central1-docker.pkg.dev"

# Cloud Storage
gcs-upload-data = "gsutil -m cp -r ./data/ gs://my-ml-bucket/datasets/"
gcs-download-checkpoint = "gsutil cp gs://my-ml-bucket/checkpoints/latest.pt ./checkpoints/"
gcs-sync-outputs = "gsutil -m rsync -r ./outputs/ gs://my-ml-bucket/runs/"

# Docker — build and push
docker-build-train = """docker build \
    --tag us-central1-docker.pkg.dev/my-project/ml-images/training:latest \
    --target training ."""
docker-build-inference = """docker build \
    --tag us-central1-docker.pkg.dev/my-project/ml-images/inference:latest \
    --target inference ."""
docker-push-train = "docker push us-central1-docker.pkg.dev/my-project/ml-images/training:latest"
docker-push-inference = "docker push us-central1-docker.pkg.dev/my-project/ml-images/inference:latest"

# Vertex AI
vertex-submit = """python -c "
from src.gcp import submit_training_job
submit_training_job(
    project='my-project',
    location='us-central1',
    display_name='training-run',
    container_uri='us-central1-docker.pkg.dev/my-project/ml-images/training:latest',
    args=['--config=configs/train.yaml'],
)"
"""
vertex-list-jobs = "gcloud ai custom-jobs list --region=us-central1 --limit=10"
vertex-logs = "gcloud ai custom-jobs stream-logs --region=us-central1"
```

## Best Practices

1. **Use Artifact Registry, not Container Registry** -- Container Registry is deprecated; Artifact Registry supports Docker, Python, and npm packages in one service.
2. **Pin image tags for Vertex AI jobs** -- never use `:latest` in production training jobs; use semantic version tags or Git SHAs.
3. **Use Workload Identity Federation** -- avoid long-lived service account keys; use OIDC tokens from GitHub Actions or GKE workloads.
4. **Store large datasets in Cloud Storage, not in Docker images** -- mount buckets with gcsfuse or download at job start.
5. **Set `staging_bucket` for Vertex AI** -- Vertex AI needs a GCS bucket for staging scripts and intermediate artifacts.
6. **Use regional resources** -- keep Artifact Registry, Cloud Storage, and Vertex AI jobs in the same region to minimize egress costs and latency.
7. **Configure lifecycle rules on GCS buckets** -- auto-delete old checkpoints and temporary outputs to control storage costs.
8. **Use prebuilt Vertex AI containers when possible** -- Google's PyTorch/TF containers have optimized CUDA and NCCL setups.
9. **Tag images with both version and `latest`** -- version tags for reproducibility, `latest` for development convenience.
10. **Grant least-privilege IAM roles** -- `roles/aiplatform.user` for submitting jobs, `roles/storage.objectViewer` for read-only data access.

## Anti-Patterns to Avoid

- ❌ Using Container Registry (`gcr.io/`) for new projects -- use Artifact Registry (`pkg.dev/`) instead.
- ❌ Baking credentials or service account keys into Docker images -- pass via environment variables or Workload Identity.
- ❌ Running Vertex AI jobs with the default Compute Engine service account -- create dedicated service accounts with minimal permissions.
- ❌ Storing training datasets inside Docker images -- images become massive and slow to pull; mount from GCS instead.
- ❌ Using `gcloud auth print-access-token` in scripts -- tokens expire after 1 hour; use `gcloud auth application-default login` or service account impersonation.
- ❌ Submitting Vertex AI jobs without a staging bucket -- the job will fail or use an auto-created bucket you cannot control.
- ❌ Using multi-region GCS buckets for training data accessed from a single region -- pay extra egress with no benefit; use regional buckets.
- ❌ Hardcoding project IDs and regions -- use Pydantic config models or environment variables for portability.
