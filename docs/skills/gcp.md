# GCP

The GCP skill covers Google Cloud Platform services for CV/ML projects: Artifact Registry, Cloud Storage, Vertex AI training jobs, and Docker image management.

**Skill directory:** `skills/gcp/`

## Purpose

ML projects running on GCP need to coordinate several services: Artifact Registry stores Docker images and Python packages, Cloud Storage holds datasets and checkpoints, and Vertex AI runs GPU-accelerated training jobs. This skill encodes the authentication flows, IAM configurations, and service patterns that keep these pieces working together reliably -- without expired tokens, overprivileged accounts, or hardcoded project IDs.

## When to Use

- Submitting training jobs to Vertex AI with custom or prebuilt containers
- Storing datasets and model artifacts in Cloud Storage
- Pushing Docker images to Artifact Registry for Vertex AI or GKE
- Setting up Workload Identity Federation for keyless CI/CD
- Defining typed GCP configuration with Pydantic models

## Key Patterns

### Submit a Vertex AI Training Job

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="resnet50-finetune",
    container_uri="us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0",
    command=["python", "-m", "my_project.train"],
)

model = job.run(
    args=["--config=configs/finetune.yaml"],
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    base_output_dir="gs://my-ml-bucket/vertex-outputs/",
)
```

### Build and Push to Artifact Registry

```bash
# Configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push
docker build --tag us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0 .
docker push us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0
```

### Typed GCP Configuration

```python
from pydantic import BaseModel, Field


class GCPConfig(BaseModel, frozen=True):
    project_id: str
    region: str = Field(default="us-central1")

class VertexJobConfig(BaseModel, frozen=True):
    display_name: str
    machine_type: str = Field(default="n1-standard-8")
    accelerator_type: str = Field(default="NVIDIA_TESLA_T4")
    accelerator_count: int = Field(default=1, ge=1, le=8)
```

## Anti-Patterns to Avoid

- Do not use deprecated Container Registry (`gcr.io/`) -- use Artifact Registry (`pkg.dev/`)
- Do not bake credentials or service account keys into Docker images
- Do not run Vertex AI jobs with the default Compute Engine service account
- Do not store training datasets inside Docker images -- mount from Cloud Storage
- Do not hardcode project IDs and regions -- use config models or environment variables

## Combines Well With

- **Docker CV** -- Dockerfile patterns that get pushed to Artifact Registry
- **Pixi** -- pixi tasks wrap gcloud and gsutil commands
- **GitHub Actions** -- CI authenticates to GCP via Workload Identity Federation
- **PyTorch Lightning** -- training modules execute inside Vertex AI jobs

## Full Reference

See [`skills/gcp/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/gcp/SKILL.md) for Artifact Registry setup, Cloud Storage patterns, Vertex AI GPU reference, authentication flows, Pydantic config models, and pixi task definitions.
