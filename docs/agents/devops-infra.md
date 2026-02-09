# DevOps/Infrastructure Agent

The DevOps/Infrastructure Agent guides all infrastructure and deployment decisions for ML/CV projects.

**Agent directory:** `agents/devops-infra/`

## Purpose

This agent provides expert advice on packaging, deploying, and operating ML services in production. It covers Docker multi-stage builds, Kubernetes deployments with GPU support, CI/CD pipeline design, monitoring and observability, infrastructure as code, and security hardening.

## Strictness Level

**ADVISORY** — This agent guides architectural decisions but does not block.

## When to Use

- Containerizing an ML model for deployment
- Choosing between deployment targets (Cloud Run, ECS, Kubernetes)
- Designing CI/CD pipelines for ML projects
- Setting up monitoring and alerting for model serving
- Writing infrastructure as code (Terraform, Kubernetes manifests)
- Reviewing security posture of ML infrastructure

## Decision Framework

The agent uses a structured decision tree:

```
Need to package an ML application?
├── Single model serving → Dockerfile with multi-stage build
├── Multiple services → Docker Compose (local), K8s (prod)
├── GPU required?
│   ├── Training → NVIDIA base images
│   └── Inference → Optimized runtime images
└── No GPU → Python slim base image
```

## Key Patterns

### Multi-Stage Docker Build

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
RUN useradd --create-home appuser
USER appuser
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment with GPU

```yaml
spec:
  containers:
    - name: api
      resources:
        requests:
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
      livenessProbe:
        httpGet:
          path: /health
      readinessProbe:
        httpGet:
          path: /ready
```

## Security Checklist

1. No secrets in code or images
2. Non-root container user
3. Minimal base images, scanned with Trivy
4. Resource limits on all containers
5. Image version pinning with content hashes

## Related Skills

- `docker-cv` — Detailed Docker patterns for CV workloads
- `github-actions` — CI/CD workflow definitions
- `gcp` — Google Cloud Platform deployment patterns
- `aws-sagemaker` — AWS managed ML infrastructure
- `fastapi` — The application framework behind deployed services

## Full Reference

See [`agents/devops-infra/SKILL.md`](https://github.com/ortizeg/whet/blob/main/agents/devops-infra/SKILL.md) for complete patterns including Docker Compose, Kubernetes HPA, Terraform, and monitoring.
