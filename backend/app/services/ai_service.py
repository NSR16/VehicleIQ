"""Reusable Gemini image-analysis operation."""

from collections.abc import Sequence
from typing import Any, Protocol

from google.genai import errors, types

from app.config import GEMINI_MODEL
from app.errors import GeminiServiceError, InvalidImageError
from app.gemini_client import get_gemini_client
from app.schemas.analysis import ImageAnalysisResponse


MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
DEFAULT_IMAGE_PROMPT = (
    "Identify the dashboard warning symbol in this image and briefly explain "
    "what it commonly indicates. If uncertain, say so."
)


class GeminiModelsClient(Protocol):
    """Minimal client surface used by this service and its test fakes."""

    def generate_content(self, *, model: str, contents: list[Any]) -> Any: ...


class GeminiClient(Protocol):
    models: GeminiModelsClient


def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    prompt: str = DEFAULT_IMAGE_PROMPT,
    *,
    client: GeminiClient | None = None,
) -> ImageAnalysisResponse:
    """Send one image and prompt to Gemini and return its text with model metadata."""
    if not image_bytes:
        raise InvalidImageError("Image bytes must not be empty.")
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise InvalidImageError(f"Unsupported image MIME type. Supported types: {supported}.")
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise InvalidImageError("Image exceeds the maximum size of 10 MB.")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return analyze_content([prompt, image_part], client=client)


def analyze_content(
    contents: Sequence[str | types.Part],
    *,
    client: GeminiClient | None = None,
) -> ImageAnalysisResponse:
    """Send text and image parts through the shared Gemini service."""
    if not contents:
        raise ValueError("Gemini content must not be empty.")
    if any(not isinstance(part, (str, types.Part)) for part in contents):
        raise TypeError("Gemini content parts must be text or google-genai Part values.")
    if any(isinstance(part, str) and not part.strip() for part in contents):
        raise ValueError("Gemini text content parts must not be empty.")

    active_client = client if client is not None else get_gemini_client()
    model = GEMINI_MODEL

    try:
        response = active_client.models.generate_content(
            model=model,
            contents=list(contents),
        )
    except errors.APIError as exc:
        raise GeminiServiceError("Gemini image analysis request failed.") from exc
    except Exception as exc:
        raise GeminiServiceError("Gemini image analysis could not be completed.") from exc

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiServiceError("Gemini returned an empty analysis response.")
    return ImageAnalysisResponse(text=text, model=model)
