# GCP Skill

The GCP Skill covers Google Cloud Platform services used in CV/ML project workflows: Artifact Registry for Docker images and Python packages, Cloud Storage for datasets and model artifacts, Vertex AI for submitting GPU-accelerated training jobs, and the Docker image lifecycle from build to push to deployment. It provides Pydantic configuration models for type-safe GCP settings and pixi task definitions that standardize common GCP operations across the team.

GCP is the cloud backbone for projects that need managed GPU training (Vertex AI), scalable artifact storage (Cloud Storage), and a private container/package registry (Artifact Registry). This skill encodes the authentication patterns, IAM best practices, and service configurations that prevent the common pitfalls of working with GCP in ML projects -- expired tokens, overprivileged service accounts, hardcoded project IDs, and bloated Docker images with baked-in datasets.

## When to Use

- When deploying training jobs to Vertex AI with custom containers and GPU configurations.
- When storing and retrieving datasets, model checkpoints, or training outputs in Cloud Storage.
- When pushing Docker images to Artifact Registry for use in Vertex AI, GKE, or Cloud Run.
- When setting up CI/CD pipelines that authenticate to GCP using Workload Identity Federation.
- When configuring GCP project settings with type-safe Pydantic models.

## Key Features

- **Artifact Registry** -- create Docker and Python package repositories, push/pull images with proper authentication, configure pip indexes.
- **Cloud Storage** -- upload datasets, download checkpoints, sync outputs with gsutil, mount buckets with gcsfuse for large-scale data access.
- **Vertex AI Training** -- submit custom training jobs with GPU selection, use prebuilt or custom containers, configure machine types and accelerators.
- **Docker image lifecycle** -- multi-stage builds for training and inference, tagging conventions, GitHub Actions integration for build-and-push workflows.
- **Authentication patterns** -- service account setup, Workload Identity Federation for keyless CI, Docker credential configuration.
- **Pydantic configuration** -- frozen config models for GCP project, Artifact Registry, Cloud Storage, and Vertex AI job specifications.
- **Pixi integration** -- task definitions for GCP auth, gsutil operations, Docker builds, and Vertex AI job submission.

## Related Skills

- **[Docker CV](../docker-cv/)** -- multi-stage Dockerfile patterns that are built and pushed to Artifact Registry by the GCP skill.
- **[Pixi](../pixi/)** -- pixi tasks wrap GCP CLI commands for consistent invocation across the team.
- **[GitHub Actions](../github-actions/)** -- CI workflows authenticate to GCP via Workload Identity and push images to Artifact Registry.
- **[PyTorch Lightning](../pytorch-lightning/)** -- training modules run inside Vertex AI custom training jobs with GPU accelerators.
