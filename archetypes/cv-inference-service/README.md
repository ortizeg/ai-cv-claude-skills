# CV Inference Service Archetype

A production-ready project template for deploying trained computer vision models as REST APIs using FastAPI and ONNX Runtime. This archetype provides everything needed to serve models behind a scalable, containerized HTTP interface with health checks, request validation, structured logging, and GPU acceleration support.

## Purpose

The CV Inference Service archetype bridges the gap between a trained model checkpoint and a deployable prediction service. In most ML teams, the transition from a Jupyter notebook or training script to a production API is where projects stall. This archetype eliminates that friction by providing a fully structured FastAPI application with ONNX Runtime inference, Docker packaging with optional CUDA support, and production-grade observability hooks.

The service is designed around ONNX Runtime as the inference backend, which decouples the serving infrastructure from the training framework. Whether you trained your model in PyTorch, TensorFlow, or JAX, as long as you can export to ONNX format, this service can serve it. ONNX Runtime also provides significant latency improvements over native framework inference through graph optimization and hardware-specific acceleration.

## Use Cases

- **Real-time inference APIs** -- Serve object detection, classification, or segmentation models behind low-latency HTTP endpoints for web and mobile applications.
- **Batch processing services** -- Accept batches of images in a single request for high-throughput offline processing pipelines.
- **Model A/B testing** -- Run multiple model versions simultaneously behind different endpoints to compare performance in production.
- **Edge deployment** -- Build minimal Docker images that run on edge devices or IoT gateways with CPU-only inference.
- **Microservice architectures** -- Deploy as one component in a larger system, communicating via REST or gRPC with upstream and downstream services.
- **Model gateway** -- Front multiple specialized models behind a unified API that routes requests based on input characteristics.

## Directory Structure

```
{{project_slug}}/
├── .github/
│   └── workflows/
│       ├── build.yml                # Docker build and push
│       ├── test.yml                 # API test pipeline
│       └── code-review.yml         # Automated code review
├── .gitignore
├── .pre-commit-config.yaml
├── .env.example                     # Environment variable template
├── pixi.toml
├── pyproject.toml
├── README.md
├── Dockerfile                       # Multi-stage production image
├── Dockerfile.gpu                   # CUDA-enabled image
├── docker-compose.yml               # Local development stack
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── py.typed
│   ├── app.py                       # FastAPI application factory
│   ├── config.py                    # Settings via pydantic-settings
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py               # Health and readiness probes
│   │   ├── predict.py              # Prediction endpoints
│   │   └── models.py               # Model management endpoints
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py               # ONNX Runtime session manager
│   │   ├── preprocessing.py        # Input normalization/resizing
│   │   └── postprocessing.py       # Output decoding (NMS, etc.)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py              # Pydantic request models
│   │   └── response.py            # Pydantic response models
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── logging.py             # Structured request logging
│   │   └── metrics.py             # Prometheus metrics
│   └── utils/
│       ├── __init__.py
│       └── image.py                # Image decode/encode helpers
├── models/
│   └── .gitkeep                     # ONNX model storage
├── scripts/
│   ├── download_model.py           # Fetch models from registry
│   └── benchmark.py                # Latency benchmarking
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test client fixtures
│   ├── test_health.py
│   ├── test_predict.py
│   └── test_preprocessing.py
└── k8s/                             # Kubernetes manifests (optional)
    ├── deployment.yaml
    ├── service.yaml
    └── hpa.yaml                    # Horizontal Pod Autoscaler
```

## Key Features

- **FastAPI** with async request handling for high concurrency under I/O-bound workloads.
- **ONNX Runtime** for framework-agnostic inference with graph optimization and hardware acceleration.
- **Pydantic v2** request/response schemas with automatic OpenAPI documentation generation.
- **Multi-stage Docker builds** producing minimal production images under 500MB for CPU and properly layered CUDA images for GPU.
- **Health and readiness probes** compatible with Kubernetes liveness and readiness checks.
- **Structured JSON logging** for integration with log aggregation systems (ELK, Datadog, CloudWatch).
- **Prometheus metrics** middleware for request latency, throughput, and error rate monitoring.
- **Horizontal scaling** with Kubernetes HPA manifests and stateless service design.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe -- returns 200 if the process is running |
| GET | `/ready` | Readiness probe -- returns 200 if models are loaded and warm |
| POST | `/predict` | Single image prediction |
| POST | `/predict/batch` | Batch prediction (multiple images) |
| GET | `/models` | List loaded models and their metadata |
| GET | `/docs` | Interactive Swagger UI documentation |

## Configuration and Environment Variables

Configuration is managed through `pydantic-settings`, which reads from environment variables with an optional `.env` file fallback. All settings are validated at startup.

| Variable | Description | Default |
|---|---|---|
| `APP_HOST` | Bind address | `0.0.0.0` |
| `APP_PORT` | Listen port | `8000` |
| `APP_WORKERS` | Uvicorn worker count | `1` |
| `MODEL_PATH` | Path to the ONNX model file | `models/model.onnx` |
| `MODEL_NAME` | Logical model name for API responses | `default` |
| `DEVICE` | Inference device (`cpu` or `cuda`) | `cpu` |
| `MAX_BATCH_SIZE` | Maximum images per batch request | `32` |
| `INPUT_SIZE` | Expected input dimensions (HxW) | `640x640` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detections | `0.5` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `CORS_ORIGINS` | Comma-separated allowed CORS origins | `*` |

## Dependencies

```toml
[dependencies]
python = ">=3.11"
fastapi = ">=0.110"
uvicorn = ">=0.27"
onnxruntime = ">=1.17"
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
pillow = ">=10.0"
numpy = ">=1.26"
python-multipart = ">=0.0.6"
```

## Usage

### Local Development

```bash
# Install dependencies
pixi install

# Copy environment template
cp .env.example .env

# Place your ONNX model in models/
cp /path/to/your/model.onnx models/model.onnx

# Start the development server with hot reload
pixi run uvicorn src.{{package_name}}.app:create_app --factory --reload --port 8000
```

### Docker Deployment

```bash
# Build CPU image
docker build -t {{project_slug}}:latest .

# Build GPU image
docker build -f Dockerfile.gpu -t {{project_slug}}:latest-gpu .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models {{project_slug}}:latest

# Run with GPU
docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models {{project_slug}}:latest-gpu

# Use docker-compose for local development
docker compose up --build
```

### Making Predictions

```bash
# Single image prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"

# Check health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready
```

### Benchmarking

```bash
# Run latency benchmark
pixi run python scripts/benchmark.py --model models/model.onnx --iterations 1000

# Profile with different batch sizes
pixi run python scripts/benchmark.py --model models/model.onnx --batch-sizes 1,4,8,16,32
```

## Customization Guide

### Adding Custom Preprocessing

Edit `src/{{package_name}}/inference/preprocessing.py` to define your input pipeline. The preprocessing module should accept raw image bytes or PIL images and return numpy arrays matching the model's expected input shape and dtype. Common operations include resizing, normalization (ImageNet mean/std or custom), color space conversion, and padding.

### Adding Custom Postprocessing

Edit `src/{{package_name}}/inference/postprocessing.py` for task-specific output decoding. For object detection, this includes non-maximum suppression and bounding box format conversion. For segmentation, this includes argmax decoding and contour extraction. For classification, this includes softmax and top-k label mapping.

### Adding New Endpoints

1. Create a new route module in `src/{{package_name}}/routes/`.
2. Define Pydantic request and response schemas in `src/{{package_name}}/schemas/`.
3. Register the router in `app.py`.
4. Add corresponding tests in `tests/`.

### Scaling Considerations

For CPU inference, scale horizontally by increasing `APP_WORKERS` or running multiple container replicas behind a load balancer. For GPU inference, use one worker per GPU and scale by adding GPU nodes. The Kubernetes HPA manifest in `k8s/hpa.yaml` is preconfigured to scale on CPU utilization and can be extended with custom metrics from the Prometheus middleware.

### Switching Inference Backends

While ONNX Runtime is the default, the engine abstraction in `inference/engine.py` can be swapped for TensorRT, OpenVINO, or TorchScript by implementing the same interface. The rest of the application remains unchanged because preprocessing, postprocessing, and routing are decoupled from the inference backend.
