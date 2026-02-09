---
name: fastapi
description: >
  FastAPI patterns for building ML model serving APIs. Covers async endpoints,
  Pydantic request/response models, dependency injection, middleware, CORS,
  background tasks, WebSocket streaming, health checks, and structured error handling.
---

# FastAPI Skill

You are building FastAPI applications for serving ML models and CV pipelines. Follow these patterns exactly.

## Core Philosophy

FastAPI provides automatic OpenAPI documentation, request validation via Pydantic, and async-first design. Every model serving endpoint in this framework uses FastAPI. Use Pydantic models for all request and response schemas — never accept raw dicts from API consumers.

## Application Structure

### Standard Application Factory

Use an application factory pattern to configure the FastAPI app. This enables testing with different configurations and clean startup/shutdown lifecycle management.

```python
"""FastAPI application factory for ML model serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field


class AppConfig(BaseModel, frozen=True):
    """Application configuration."""

    title: str = "ML Model API"
    version: str = "1.0.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    model_path: str = "models/best.onnx"
    max_batch_size: int = 32
    device: str = "cuda:0"


class ModelRegistry:
    """Holds loaded models for the application lifetime."""

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}

    async def load(self, config: AppConfig) -> None:
        logger.info("Loading model from {}", config.model_path)
        # Load ONNX, TensorRT, or PyTorch model here
        self.models["default"] = await _load_model(config.model_path)
        logger.info("Model loaded successfully")

    async def shutdown(self) -> None:
        logger.info("Releasing model resources")
        self.models.clear()


model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage model loading on startup and cleanup on shutdown."""
    config = AppConfig()
    await model_registry.load(config)
    yield
    await model_registry.shutdown()


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or AppConfig()

    app = FastAPI(
        title=config.title,
        version=config.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes import prediction, health
    app.include_router(health.router, tags=["health"])
    app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])

    return app
```

## Request and Response Models

### Pydantic Schemas for ML Endpoints

Define explicit Pydantic models for every endpoint. Never use `dict` or `Any` in API signatures.

```python
"""Pydantic schemas for prediction endpoints."""

from __future__ import annotations

import base64

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel, frozen=True):
    """Single image prediction request."""

    image_b64: str = Field(..., description="Base64-encoded image bytes")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence for detections"
    )
    max_detections: int = Field(default=100, ge=1, le=1000)

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 encoding") from exc
        return v


class Detection(BaseModel, frozen=True):
    """Single object detection result."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2]")


class PredictionResponse(BaseModel, frozen=True):
    """Prediction response with detections and metadata."""

    detections: list[Detection]
    inference_time_ms: float
    model_version: str


class BatchPredictionRequest(BaseModel, frozen=True):
    """Batch prediction request."""

    images: list[PredictionRequest] = Field(max_length=32)


class BatchPredictionResponse(BaseModel, frozen=True):
    """Batch prediction response."""

    results: list[PredictionResponse]
    total_inference_time_ms: float


class ErrorResponse(BaseModel, frozen=True):
    """Structured error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None
```

## Endpoint Patterns

### Prediction Endpoint with Dependency Injection

```python
"""Prediction routes."""

from __future__ import annotations

import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    Detection,
)
from .dependencies import get_model, get_preprocessor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: Annotated[Model, Depends(get_model)],
    preprocessor: Annotated[Preprocessor, Depends(get_preprocessor)],
) -> PredictionResponse:
    """Run inference on a single image."""
    start = time.perf_counter()

    image = preprocessor.decode_and_preprocess(request.image_b64)
    raw_detections = await model.predict(image)

    detections = [
        Detection(label=d.label, confidence=d.confidence, bbox=d.bbox)
        for d in raw_detections
        if d.confidence >= request.confidence_threshold
    ][: request.max_detections]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        detections=detections,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=model.version,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model: Annotated[Model, Depends(get_model)],
    preprocessor: Annotated[Preprocessor, Depends(get_preprocessor)],
) -> BatchPredictionResponse:
    """Run inference on a batch of images."""
    start = time.perf_counter()

    results: list[PredictionResponse] = []
    for item in request.images:
        result = await predict(item, model, preprocessor)
        results.append(result)

    total_ms = (time.perf_counter() - start) * 1000

    return BatchPredictionResponse(
        results=results,
        total_inference_time_ms=round(total_ms, 2),
    )
```

### Health Check Endpoints

```python
"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel, frozen=True):
    status: str
    model_loaded: bool
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness and readiness probe."""
    from .app import model_registry

    return HealthResponse(
        status="healthy",
        model_loaded=bool(model_registry.models),
        version="1.0.0",
    )


@router.get("/ready")
async def readiness() -> dict[str, bool]:
    """Kubernetes readiness probe."""
    from .app import model_registry

    if not model_registry.models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}
```

## Dependency Injection

### Model and Preprocessor Dependencies

```python
"""FastAPI dependencies for ML serving."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Load application configuration once."""
    return AppConfig()


async def get_model(
    config: AppConfig = Depends(get_app_config),
) -> Model:
    """Provide the loaded model instance."""
    from .app import model_registry

    model = model_registry.models.get("default")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


async def get_preprocessor(
    config: AppConfig = Depends(get_app_config),
) -> Preprocessor:
    """Provide the image preprocessor."""
    return Preprocessor(target_size=(640, 640), device=config.device)
```

## Middleware and Error Handling

### Request ID and Logging Middleware

```python
"""Custom middleware for ML API."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "{method} {path} → {status} ({elapsed:.1f}ms) [{rid}]",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed=elapsed,
            rid=request_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response
```

### Structured Exception Handlers

```python
"""Exception handlers for consistent error responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger


async def model_error_handler(request: Request, exc: ModelInferenceError) -> JSONResponse:
    """Handle model inference failures."""
    logger.error("Inference error: {}", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "inference_error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def validation_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle input validation errors with clear messages."""
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": str(exc)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    app.add_exception_handler(ModelInferenceError, model_error_handler)
```

## WebSocket Streaming

### Real-Time Video Inference

```python
"""WebSocket endpoint for streaming inference."""

from __future__ import annotations

import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


@router.websocket("/ws/stream")
async def stream_inference(websocket: WebSocket) -> None:
    """Stream video frames and return detections in real time."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            if not frame_b64:
                await websocket.send_json({"error": "Missing 'frame' field"})
                continue

            frame_bytes = base64.b64decode(frame_b64)
            detections = await run_inference(frame_bytes)

            await websocket.send_json({
                "detections": [d.model_dump() for d in detections],
                "frame_id": data.get("frame_id"),
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
```

## Background Tasks

### Async Post-Processing

```python
"""Background task patterns for FastAPI ML APIs."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from loguru import logger

router = APIRouter()


async def save_prediction_to_db(
    request_id: str,
    detections: list[Detection],
) -> None:
    """Save prediction results asynchronously after response."""
    logger.info("Saving {} detections for request {}", len(detections), request_id)
    await db.predictions.insert_one({
        "request_id": request_id,
        "detections": [d.model_dump() for d in detections],
    })


@router.post("/predict")
async def predict_with_logging(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """Predict and log results in background."""
    result = await run_prediction(request)

    background_tasks.add_task(
        save_prediction_to_db,
        request_id=request.state.request_id,
        detections=result.detections,
    )

    return result
```

## Testing FastAPI Applications

### Async Test Client

```python
"""Tests for prediction API."""

from __future__ import annotations

import base64

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client() -> AsyncClient:
    """Create async test client."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_predict_returns_detections(client: AsyncClient) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    request_body = {
        "image_b64": base64.b64encode(image_bytes).decode(),
        "confidence_threshold": 0.5,
    }
    response = await client.post("/api/v1/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "inference_time_ms" in data


@pytest.mark.anyio
async def test_predict_rejects_invalid_base64(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/predict",
        json={"image_b64": "not-valid-base64!!!"},
    )
    assert response.status_code == 422
```

## Docker Deployment

### Production Dockerfile for FastAPI ML API

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Uvicorn Configuration

```python
"""Uvicorn runner with production settings."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True,
        limit_concurrency=100,
        timeout_keep_alive=30,
    )
```

## Anti-Patterns

- **Never use `dict` for request/response models** — always define Pydantic schemas with explicit fields and validators.
- **Never load models inside endpoint functions** — use the lifespan context manager for startup/shutdown lifecycle.
- **Never block the event loop with synchronous inference** — use `await` with async model wrappers or `run_in_executor` for CPU-bound operations.
- **Never hardcode CORS origins in production** — load from configuration.
- **Never return raw numpy arrays or tensors** — serialize to lists or base64 in the response model.
- **Never skip input validation** — use Pydantic field validators for base64, image dimensions, and parameter bounds.

## Integration with Other Skills

- **Pydantic Strict** — All request/response models follow frozen BaseModel patterns.
- **Docker CV** — Production Dockerfiles with multi-stage builds for FastAPI + model serving.
- **ONNX / TensorRT** — Load optimized models in the lifespan handler.
- **Loguru** — Structured logging in middleware and exception handlers.
- **Testing** — Async test client with httpx for full endpoint coverage.
