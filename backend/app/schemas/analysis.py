"""Minimal input and output schemas for image analysis."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class Strategy(str, Enum):
    ordinary = "ordinary"
    contrastive = "contrastive"


class ImageStatus(str, Enum):
    suitable = "suitable"
    unclear_image = "unclear_image"
    irrelevant_image = "irrelevant_image"
    no_indicator_visible = "no_indicator_visible"


class DiagnosisStatus(str, Enum):
    ok = "ok"
    needs_clarification = "needs_clarification"
    unclear_image = "unclear_image"
    irrelevant_image = "irrelevant_image"
    no_indicator_visible = "no_indicator_visible"
    unknown = "unknown"


class ClarificationQuestion(BaseModel):
    id: str
    question: str
    optional: bool = True


class IndicatorResult(BaseModel):
    label_id: str
    display_name: str
    confidence: Literal["low", "medium", "high"]
    visible_colour: str | None = None
    state: Literal["steady", "flashing", "unknown"] = "unknown"
    visual_evidence: str = Field(min_length=1, max_length=500)
    meaning: str
    severity: str
    recommended_action: str


class DiagnosisMetadata(BaseModel):
    strategy: Strategy
    model: str
    prompt_version: str = "1.0"
    temperature: float = 0.0
    latency_ms: int = Field(ge=0)
    few_shot_examples_used: bool = False
    few_shot_example_count: int = Field(default=0, ge=0)
    demonstration_ids: list[str] = Field(default_factory=list)
    prompt_mode: Literal["suitability_only", "ordinary_few_shot", "contrastive_few_shot"] = "suitability_only"


class DiagnosisResponse(BaseModel):
    status: DiagnosisStatus
    image_status: ImageStatus
    indicator: IndicatorResult | None = None
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    message: str
    safety_disclaimer: str = (
        "VehicleIQ provides preliminary informational assistance and is not a substitute "
        "for a qualified mechanic or your vehicle owner's manual."
    )
    metadata: DiagnosisMetadata

    @model_validator(mode="after")
    def status_matches_payload(self):
        if self.status == DiagnosisStatus.ok:
            if self.image_status != ImageStatus.suitable or self.indicator is None:
                raise ValueError("ok response requires a suitable image and indicator")
            if self.clarification_questions:
                raise ValueError("ok response cannot include clarification questions")
        elif self.indicator is not None:
            raise ValueError("abstention and clarification responses cannot include an indicator")
        if self.status == DiagnosisStatus.needs_clarification and not self.clarification_questions:
            raise ValueError("needs_clarification response requires at least one question")
        if self.status != DiagnosisStatus.needs_clarification and self.clarification_questions:
            raise ValueError("only needs_clarification response may include questions")
        return self
