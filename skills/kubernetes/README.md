# Kubernetes Skill

The Kubernetes Skill provides production-grade deployment patterns for ML inference and training services on Kubernetes clusters. ML workloads have unique orchestration requirements: GPU scheduling with NVIDIA device plugins, long startup times for model loading, large persistent volumes for model weights, autoscaling based on inference throughput, and environment-specific configurations across dev, staging, and production. This skill addresses all of these with opinionated manifests, Helm charts, Kustomize overlays, and operational best practices tailored to computer vision and deep learning workloads.

The skill covers the full deployment lifecycle: Deployment manifests with GPU resource requests and tolerations, Services and Ingress resources for routing traffic, Horizontal Pod Autoscalers tuned for inference workloads, ConfigMaps and Secrets for model configuration, PersistentVolumeClaims and init containers for model storage, Kubernetes Jobs for training runs, startup/liveness/readiness probes calibrated for slow model loading, namespace organization with resource quotas, and Kustomize overlays for managing environment differences without duplicating YAML.

## Purpose

When you need to deploy ML models to Kubernetes -- whether for real-time inference APIs, batch processing jobs, or GPU-accelerated training -- this skill provides the patterns and manifests to do it correctly. It encodes best practices for GPU scheduling, health checking, autoscaling, and multi-environment management that are specific to ML/CV workloads.

## When to Use

- When deploying ML inference services to a Kubernetes cluster (GKE, EKS, AKS, or on-premise).
- When you need GPU-accelerated pods with proper NVIDIA device plugin configuration and resource requests.
- When building Helm charts or Kustomize overlays for ML services across multiple environments.
- When configuring autoscaling, health probes, and persistent storage for model serving workloads.
- When orchestrating training jobs on Kubernetes with multi-GPU scheduling.

## Key Features

- **GPU resource scheduling** -- proper `nvidia.com/gpu` requests and limits with node selectors and tolerations for GPU node pools.
- **Health probes for ML** -- startup, liveness, and readiness probes calibrated for containers that take minutes to load large models.
- **Horizontal Pod Autoscaler** -- scaling policies tuned for inference workloads with stabilization windows and custom metrics.
- **Helm chart structure** -- complete chart layout with values files per environment for templated deployments.
- **Kustomize overlays** -- base manifests with dev, staging, and prod overlays to avoid YAML duplication.
- **Persistent model storage** -- PVCs and init containers for downloading model weights from object storage.
- **Namespace organization** -- environment separation with resource quotas to prevent GPU monopolization.
- **Training Jobs** -- Kubernetes Job manifests for one-off or scheduled training runs with GPU resources.

## Related Skills

- **[Docker CV](../docker-cv/)** -- container images deployed to Kubernetes are built using multi-stage Docker patterns with CUDA support.
- **[FastAPI](../fastapi/)** -- inference APIs running inside Kubernetes pods use FastAPI with health endpoints for probe integration.
- **[GCP](../gcp/)** -- GKE clusters, Artifact Registry for images, and Cloud Storage for model artifacts.
- **[AWS SageMaker](../aws-sagemaker/)** -- alternative managed deployment; Kubernetes provides more control when SageMaker constraints are too rigid.
