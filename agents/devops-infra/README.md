# DevOps/Infrastructure Agent

The DevOps/Infrastructure Agent guides all infrastructure and deployment decisions for ML/CV projects, from containerization to production orchestration.

## Purpose

This agent provides expert advice on packaging, deploying, and operating ML services in production. It covers Docker multi-stage builds, Kubernetes deployments with GPU support, CI/CD pipeline design, monitoring and observability, infrastructure as code, and security hardening.

## Strictness Level

**ADVISORY** — This agent guides architectural decisions but does not block.

## When to Use

- Containerizing an ML model for deployment (Docker, multi-stage builds).
- Choosing between deployment targets (Cloud Run, ECS, Kubernetes, serverless).
- Designing CI/CD pipelines for ML projects (test, build, deploy).
- Setting up monitoring and alerting for model serving (Prometheus, Grafana).
- Writing infrastructure as code (Terraform, Kubernetes manifests).
- Reviewing security posture of ML infrastructure.

## Example Session

```
You: "I need to deploy my PyTorch model as an API with GPU support"

DevOps Agent: "I recommend this architecture:
1. Multi-stage Dockerfile with NVIDIA base image
2. FastAPI serving layer with health checks
3. Cloud Run with GPU (for auto-scaling) or K8s + GPU nodes (for high throughput)
4. Prometheus metrics for inference latency tracking
5. GitHub Actions pipeline: test → build → push → deploy"
```

## Related Skills

- `docker-cv` — Detailed Docker patterns for CV workloads
- `github-actions` — CI/CD workflow definitions
- `gcp` — Google Cloud Platform deployment patterns
- `aws-sagemaker` — AWS managed ML infrastructure
- `fastapi` — The application framework behind the deployed services
