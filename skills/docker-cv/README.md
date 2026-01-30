# Docker CV Skill

The Docker CV Skill provides multi-stage Docker build patterns specifically designed for computer vision and machine learning workloads that require CUDA support. CV/ML projects face unique containerization challenges: large base images with GPU drivers, heavy dependency trees including PyTorch and OpenCV, long build times, and the need for both heavyweight training containers and slim inference images. This skill addresses all of these with opinionated Dockerfile patterns, Docker Compose configurations with GPU passthrough, and security best practices tailored to ML workflows.

The skill defines a multi-stage build strategy: a base stage with CUDA runtime and system libraries, a dependencies stage that installs Python packages via pixi, a training stage with full development tools and debugging capabilities, and a slim inference stage that strips everything unnecessary for serving. Docker Compose files configure GPU resource reservations, shared memory sizing for DataLoader workers, volume mounts for datasets and model artifacts, and environment variable injection for experiment tracking credentials.

## When to Use

- When packaging a training pipeline for execution on GPU-equipped cloud instances or on-premise servers.
- When building a lightweight inference container for model serving or edge deployment.
- When setting up a local development environment with GPU access through Docker Compose.
- When you need reproducible builds that guarantee the same CUDA, cuDNN, and driver versions across environments.

## Key Features

- **Multi-stage builds** -- separate stages for base, dependencies, training, and inference to minimize image sizes and build times.
- **CUDA support** -- base images pinned to specific CUDA and cuDNN versions with driver compatibility documented.
- **Docker Compose GPU passthrough** -- `deploy.resources.reservations.devices` configuration for NVIDIA GPU access with configurable memory limits.
- **Shared memory configuration** -- correct `shm_size` settings to prevent DataLoader crashes with multiple workers.
- **Security best practices** -- non-root users, no secrets in layers, `.dockerignore` for datasets and checkpoints, and read-only filesystem options.
- **Layer caching strategy** -- dependency installation ordered to maximize cache hits when only source code changes.

## Related Skills

- **[Pixi](../pixi/)** -- Docker builds use pixi for dependency installation, ensuring container environments match local development exactly.
- **[PyTorch Lightning](../pytorch-lightning/)** -- training containers run Lightning Trainer with appropriate accelerator and strategy settings.
- **[GitHub Actions](../github-actions/)** -- CI pipelines build, tag, and push Docker images as part of the release workflow.
- **[Code Quality](../code-quality/)** -- Dockerfiles are linted with hadolint, integrated into the same quality gates as Python code.
