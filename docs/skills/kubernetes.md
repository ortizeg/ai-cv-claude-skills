# Kubernetes

The Kubernetes skill provides expert patterns for deploying ML services on Kubernetes, covering Deployments, Services, GPU scheduling, Helm charts, and autoscaling.

**Skill directory:** `skills/kubernetes/`

## Purpose

Kubernetes is the standard orchestration platform for production ML serving at scale. This skill encodes best practices for deploying ML models on K8s: GPU resource requests, health check probes, Horizontal Pod Autoscaler configuration, Helm chart structure, and environment management with Kustomize.

## When to Use

Use this skill whenever you need to:

- Deploy model serving APIs on Kubernetes with GPU support
- Set up autoscaling based on inference latency or GPU utilization
- Manage multi-environment deployments (dev/staging/prod)
- Package ML services as Helm charts

## Key Patterns

### GPU Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: model-api
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

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Anti-Patterns to Avoid

- Do not use `latest` image tags -- pin versions with SHA digests
- Do not skip resource limits -- unbounded pods risk node instability
- Do not store model weights in container images -- use persistent volumes

## Combines Well With

- **Docker CV** -- Container images deployed to K8s
- **FastAPI** -- The serving framework inside K8s pods
- **GCP / AWS SageMaker** -- Managed K8s clusters (GKE, EKS)

## Full Reference

See [`skills/kubernetes/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/kubernetes/SKILL.md) for complete patterns including Helm charts, Kustomize, and monitoring.
