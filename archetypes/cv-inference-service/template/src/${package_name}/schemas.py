"""Pydantic schemas for ${project_name}."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel, frozen=True):
    """Health check response."""

    status: str
    version: str


class PredictionRequest(BaseModel, frozen=True):
    """Prediction request."""

    image_b64: str = Field(..., description="Base64-encoded image")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Detection(BaseModel, frozen=True):
    """Single detection result."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel, frozen=True):
    """Prediction response."""

    detections: list[Detection]
    inference_time_ms: float
