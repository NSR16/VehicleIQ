"""Minimal input and output schemas for image analysis."""

from typing import Literal

from pydantic import BaseModel, Field


class VehicleContext(BaseModel):
    """Optional vehicle information that may accompany a future analysis request."""

    manufacturer: str
    model: str
    manufacturing_year: int
    symptoms: str | None = None


class ImageMetadata(BaseModel):
    """Image details suitable for an API request description."""

    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    size_bytes: int = Field(ge=0)
    filename: str | None = None


class ImageAnalysisRequest(BaseModel):
    """Small metadata schema for a future image-analysis request."""

    vehicle: VehicleContext
    image: ImageMetadata


class ImageAnalysisResponse(BaseModel):
    """Raw generated analysis and the model that produced it."""

    text: str
    model: str
