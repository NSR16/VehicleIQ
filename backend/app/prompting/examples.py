"""Caller-provided few-shot examples and contrastive pair representation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from app.errors import InvalidImageError
from app.services.ai_service import MAX_IMAGE_SIZE_BYTES


ImageMimeType = Literal["image/png", "image/jpeg", "image/webp"]


class FewShotExample(BaseModel):
    """One image with its supplied ground-truth label and meaning."""

    model_config = ConfigDict(strict=True)

    image_bytes: bytes
    mime_type: ImageMimeType
    label: str
    meaning: str
    example_id: str | None = None
    source_note: str | None = None

    @field_validator("image_bytes")
    @classmethod
    def validate_image_bytes(cls, value: bytes) -> bytes:
        if not value:
            raise InvalidImageError("Example image bytes must not be empty.")
        if len(value) > MAX_IMAGE_SIZE_BYTES:
            raise InvalidImageError("Example image exceeds the maximum size of 10 MB.")
        return value

    @field_validator("label", "meaning")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Example label and meaning must not be empty.")
        return value


class ContrastivePair(BaseModel):
    """Two caller-curated examples and their supplied point of distinction."""

    first: FewShotExample
    second: FewShotExample
    visual_distinction: str
    semantic_difference: str

    @field_validator("visual_distinction", "semantic_difference")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pair distinction explanations must not be empty.")
        return value


def examples_from_pairs(pairs: list[ContrastivePair]) -> list[FewShotExample]:
    """Flatten pairs in order for a baseline using exactly the same images and labels."""
    return [example for pair in pairs for example in (pair.first, pair.second)]
