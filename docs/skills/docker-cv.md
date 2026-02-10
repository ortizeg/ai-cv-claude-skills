# Docker CV

The Docker CV skill provides container patterns optimized for computer vision workloads, including GPU support, large model files, and multi-stage builds.

**Skill directory:** `skills/docker-cv/`

## Purpose

CV containers have unique requirements: CUDA runtime, large model weights, OpenCV system dependencies, and the need for both training (heavy, GPU) and inference (lean, optimized) images. This skill encodes Dockerfile patterns that handle all of these concerns with proper layer caching, non-root users, and health checks.

## When to Use

- Containerizing training pipelines for cluster execution
- Building inference service images with GPU support
- Creating reproducible development environments
- Deploying CV models to Kubernetes or cloud services

## Key Patterns

### Multi-Stage Training Container

```dockerfile
# Stage 1: Builder
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --frozen

COPY . .

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

USER appuser
ENTRYPOINT ["pixi", "run", "python", "src/my_project/train.py"]
```

### Docker Compose for Training

```yaml
services:
  train:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

## Anti-Patterns to Avoid

- Do not install the full CUDA toolkit in runtime images -- use `-runtime` base images
- Do not copy the entire project before installing dependencies -- breaks layer caching
- Do not run containers as root in production
- Do not bake model weights into the image -- mount them as volumes or download at startup

## Combines Well With

- **Pixi** -- Pixi manages dependencies inside the container
- **ONNX** -- Optimized inference containers with ONNX Runtime
- **GitHub Actions** -- Build and push Docker images in CI
- **PyTorch Lightning** -- Containerized distributed training

## Full Reference

See [`skills/docker-cv/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/docker-cv/SKILL.md) for patterns including inference-optimized images, TensorRT integration, and Kubernetes deployment manifests.
