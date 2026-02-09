---
name: kubernetes
description: >
  Kubernetes deployment patterns for ML inference and training services. Covers
  GPU resource scheduling, Helm charts, Kustomize overlays, health probes,
  autoscaling, persistent volumes for model storage, and namespace organization
  for dev/staging/prod environments.
---

# Kubernetes Skill

Kubernetes deployment patterns for ML inference and training services with GPU scheduling, Helm charts, autoscaling, and environment-specific overlays.

## Deployment Manifests for ML Services

Define Deployments with explicit resource requests and GPU scheduling for inference workloads.

### Inference Deployment

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  labels:
    app: model-server
    component: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
        component: inference
    spec:
      containers:
        - name: model-server
          image: registry.example.com/ml-images/inference:v1.2.0
          ports:
            - containerPort: 8000
              name: http
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "1"
          envFrom:
            - configMapRef:
                name: model-config
            - secretRef:
                name: model-secrets
          volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 60
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 5
          startupProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
            failureThreshold: 30
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

## GPU Resource Requests

Always set both `requests` and `limits` for `nvidia.com/gpu`. Kubernetes GPU scheduling requires exact counts -- fractional GPUs are not natively supported.

```yaml
# GPU resource patterns
resources:
  requests:
    nvidia.com/gpu: "1"   # Request exactly 1 GPU
  limits:
    nvidia.com/gpu: "1"   # Must equal requests for GPUs

# Multi-GPU training pod
resources:
  requests:
    cpu: "8"
    memory: "32Gi"
    nvidia.com/gpu: "4"
  limits:
    cpu: "16"
    memory: "64Gi"
    nvidia.com/gpu: "4"
```

### NVIDIA Device Plugin

The NVIDIA device plugin must be deployed in the cluster to enable GPU scheduling.

```bash
# Install the NVIDIA device plugin via Helm
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
    --namespace kube-system \
    --set runtimeClassName=nvidia
```

## Service and Ingress Configuration

### Service

```yaml
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
  labels:
    app: model-server
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: http
      protocol: TCP
      name: http
  selector:
    app: model-server
```

### Ingress

```yaml
# k8s/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-server
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.ml.example.com
      secretName: model-server-tls
  rules:
    - host: api.ml.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-server
                port:
                  name: http
```

## Horizontal Pod Autoscaler

Scale inference pods based on CPU, memory, or custom metrics such as request latency or GPU utilization.

```yaml
# k8s/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 10
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: inference_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

## ConfigMaps and Secrets for Model Config

### ConfigMap

```yaml
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_NAME: "resnet50"
  MODEL_VERSION: "v1.2.0"
  MODEL_PATH: "/models/resnet50_v1.2.0.onnx"
  BATCH_SIZE: "8"
  NUM_WORKERS: "4"
  LOG_LEVEL: "INFO"
  CONFIDENCE_THRESHOLD: "0.5"
```

### Secret

```yaml
# k8s/base/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-secrets
type: Opaque
stringData:
  WANDB_API_KEY: ""         # Populated via Kustomize or sealed-secrets
  S3_ACCESS_KEY: ""
  S3_SECRET_KEY: ""
```

```bash
# Create secrets from command line (never commit plaintext secrets)
kubectl create secret generic model-secrets \
    --from-literal=WANDB_API_KEY="${WANDB_API_KEY}" \
    --from-literal=S3_ACCESS_KEY="${S3_ACCESS_KEY}" \
    --from-literal=S3_SECRET_KEY="${S3_SECRET_KEY}" \
    --namespace=ml-inference
```

## Persistent Volumes for Model Storage

Use PersistentVolumeClaims to store large model weights independently of pod lifecycle.

```yaml
# k8s/base/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: standard
  resources:
    requests:
      storage: 50Gi
```

### Init Container to Download Models

```yaml
# Use an init container to pull model weights before the main container starts
spec:
  initContainers:
    - name: model-downloader
      image: google/cloud-sdk:slim
      command:
        - gsutil
        - cp
        - gs://my-ml-bucket/models/resnet50_v1.2.0.onnx
        - /models/resnet50_v1.2.0.onnx
      volumeMounts:
        - name: model-storage
          mountPath: /models
  containers:
    - name: model-server
      image: registry.example.com/ml-images/inference:v1.2.0
      volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
  volumes:
    - name: model-storage
      emptyDir:
        sizeLimit: 50Gi
```

## Helm Chart Patterns

Organize Kubernetes manifests into a Helm chart for templated, reusable deployments.

### Chart Structure

```
helm/model-server/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-staging.yaml
├── values-prod.yaml
└── templates/
    ├── _helpers.tpl
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── pvc.yaml
    └── NOTES.txt
```

### values.yaml

```yaml
# helm/model-server/values.yaml
replicaCount: 2

image:
  repository: registry.example.com/ml-images/inference
  tag: "v1.2.0"
  pullPolicy: IfNotPresent

model:
  name: resnet50
  version: v1.2.0
  path: /models/resnet50_v1.2.0.onnx
  storageSizeGi: 50

resources:
  requests:
    cpu: "2"
    memory: "4Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "4"
    memory: "8Gi"
    nvidia.com/gpu: "1"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70

ingress:
  enabled: true
  host: api.ml.example.com
  tls: true

probes:
  liveness:
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 15
  readiness:
    path: /health/ready
    initialDelaySeconds: 60
    periodSeconds: 10
  startup:
    path: /health
    initialDelaySeconds: 10
    failureThreshold: 30

nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-tesla-t4

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Helm Deployment Commands

```bash
# Install the chart
helm install model-server ./helm/model-server \
    --namespace ml-inference \
    --create-namespace \
    --values helm/model-server/values-prod.yaml

# Upgrade with new image tag
helm upgrade model-server ./helm/model-server \
    --namespace ml-inference \
    --set image.tag=v1.3.0

# Rollback to previous release
helm rollback model-server 1 --namespace ml-inference

# Dry-run to preview rendered templates
helm template model-server ./helm/model-server \
    --values helm/model-server/values-staging.yaml \
    --debug
```

## Health Checks (Liveness, Readiness, and Startup Probes)

ML containers need long startup times for model loading. Use all three probe types.

```yaml
# Startup probe: allow up to 5 minutes for model loading (10s * 30 attempts)
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 30

# Liveness probe: restart if unresponsive
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 15
  timeoutSeconds: 5
  failureThreshold: 3

# Readiness probe: remove from service if not ready
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 5
```

### FastAPI Health Endpoints

```python
from fastapi import FastAPI, Response, status

app = FastAPI()

model_loaded = False


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe -- is the process alive?"""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness(response: Response) -> dict[str, str]:
    """Readiness probe -- is the model loaded and ready for inference?"""
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "model loading"}
    return {"status": "ready"}
```

## Namespace Organization

Separate environments and workload types using namespaces with resource quotas.

```bash
# Create namespaces for each environment
kubectl create namespace ml-dev
kubectl create namespace ml-staging
kubectl create namespace ml-prod

# Create namespace for training jobs
kubectl create namespace ml-training
```

### Resource Quotas per Namespace

```yaml
# k8s/namespaces/ml-prod-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-prod
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    requests.cpu: "32"
    requests.memory: "128Gi"
    persistentvolumeclaims: "20"
```

## Kustomize Overlays for Dev/Staging/Prod

Use Kustomize to manage environment-specific variations without duplicating manifests.

### Directory Structure

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   └── pvc.yaml
└── overlays/
    ├── dev/
    │   ├── kustomization.yaml
    │   └── patches/
    │       ├── deployment-patch.yaml
    │       └── hpa-patch.yaml
    ├── staging/
    │   ├── kustomization.yaml
    │   └── patches/
    │       └── deployment-patch.yaml
    └── prod/
        ├── kustomization.yaml
        └── patches/
            ├── deployment-patch.yaml
            └── hpa-patch.yaml
```

### Base Kustomization

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - ingress.yaml
  - hpa.yaml
  - configmap.yaml
  - pvc.yaml
commonLabels:
  app.kubernetes.io/name: model-server
  app.kubernetes.io/managed-by: kustomize
```

### Dev Overlay

```yaml
# k8s/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ml-dev
resources:
  - ../../base
patches:
  - path: patches/deployment-patch.yaml
  - path: patches/hpa-patch.yaml
images:
  - name: registry.example.com/ml-images/inference
    newTag: dev-latest
```

```yaml
# k8s/overlays/dev/patches/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: model-server
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "1"
```

```yaml
# k8s/overlays/dev/patches/hpa-patch.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server
spec:
  minReplicas: 1
  maxReplicas: 2
```

### Prod Overlay

```yaml
# k8s/overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ml-prod
resources:
  - ../../base
patches:
  - path: patches/deployment-patch.yaml
  - path: patches/hpa-patch.yaml
images:
  - name: registry.example.com/ml-images/inference
    newTag: v1.2.0
```

```yaml
# k8s/overlays/prod/patches/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: model-server
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "16Gi"
              nvidia.com/gpu: "1"
```

### Apply Overlays

```bash
# Apply dev overlay
kubectl apply -k k8s/overlays/dev/

# Apply staging overlay
kubectl apply -k k8s/overlays/staging/

# Apply prod overlay
kubectl apply -k k8s/overlays/prod/

# Preview rendered manifests without applying
kubectl kustomize k8s/overlays/prod/
```

## Training Jobs with Kubernetes

Use Kubernetes Jobs for one-off training runs.

```yaml
# k8s/jobs/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-resnet50
  namespace: ml-training
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: trainer
          image: registry.example.com/ml-images/training:v1.2.0
          command: ["python", "-m", "my_project.train"]
          args:
            - "--config=configs/train.yaml"
            - "--epochs=100"
            - "--batch-size=32"
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "4"
            limits:
              cpu: "16"
              memory: "64Gi"
              nvidia.com/gpu: "4"
          volumeMounts:
            - name: data
              mountPath: /data
              readOnly: true
            - name: checkpoints
              mountPath: /checkpoints
          envFrom:
            - secretRef:
                name: training-secrets
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: training-data-pvc
        - name: checkpoints
          persistentVolumeClaim:
            claimName: checkpoints-pvc
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
```

## Anti-Patterns to Avoid

- Do not omit GPU resource limits -- without explicit `nvidia.com/gpu` limits, pods will not be scheduled on GPU nodes and will silently fall back to CPU.
- Do not use `latest` image tags in production -- always pin to a specific version tag or Git SHA for reproducibility and safe rollbacks.
- Do not skip startup probes for ML containers -- model loading can take minutes; without a startup probe, Kubernetes will kill pods before they are ready.
- Do not store model weights inside container images -- images become multi-gigabyte and slow to pull; use PVCs or init containers to download models.
- Do not set CPU requests too low for GPU pods -- GPU inference still requires CPU for preprocessing; under-provisioned CPU starves the pipeline.
- Do not hardcode environment-specific values in base manifests -- use Kustomize overlays or Helm values files for dev/staging/prod differences.
- Do not run pods as root -- always set `runAsNonRoot: true` in the pod security context.
- Do not skip resource quotas on shared clusters -- without quotas, a single team can monopolize all GPU resources.
- Do not expose inference services without TLS -- always terminate TLS at the Ingress or use a service mesh.
- Do not ignore pod disruption budgets -- set `minAvailable` to prevent all replicas from being evicted during node upgrades.

## Best Practices

1. **Always set resource requests and limits** -- GPU, CPU, and memory must all be specified for predictable scheduling.
2. **Use startup probes** -- ML containers need extended initialization time; startup probes prevent premature restarts.
3. **Separate namespaces by environment** -- dev, staging, and prod with resource quotas per namespace.
4. **Use Kustomize or Helm for environment management** -- never duplicate YAML across environments.
5. **Pin image tags** -- use semantic versions or Git SHAs, never `latest` in staging or prod.
6. **Run as non-root** -- set `securityContext.runAsNonRoot: true` on all pods.
7. **Use PVCs for model storage** -- decouple model weights from container images.
8. **Set pod disruption budgets** -- ensure minimum availability during cluster maintenance.
9. **Configure HPA with appropriate metrics** -- scale on request rate or GPU utilization, not just CPU.
10. **Use init containers for model downloads** -- pull weights from object storage before the main container starts.
