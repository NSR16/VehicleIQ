"""Reusable Gemini image-analysis operation."""

from collections.abc import Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
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


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


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
    config: types.GenerateContentConfig | None = None,
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
            config=config or types.GenerateContentConfig(temperature=0.0),
        )
    except errors.APIError as exc:
        raise GeminiServiceError(
            "Gemini image analysis request failed.",
            status_code=getattr(exc, "code", None),
            provider_status=getattr(exc, "status", None),
            provider_message=getattr(exc, "message", None),
            provider_details=getattr(exc, "details", None),
            retry_after_seconds=_retry_after_seconds(getattr(exc, "response", None)),
            retryable=(getattr(exc, "code", None) in {408, 500, 502, 503, 504}),
        ) from exc
    except Exception as exc:
        transient_names = ("Timeout", "ConnectError", "NetworkError", "RemoteProtocolError")
        retryable = isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__.endswith(transient_names)
        raise GeminiServiceError(
            "Gemini image analysis could not be completed.", retryable=retryable
        ) from exc

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiServiceError("Gemini returned an empty analysis response.")
    usage = getattr(response, "usage_metadata", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage_data = usage.model_dump(exclude_none=True)
    elif usage is not None and hasattr(usage, "__dict__"):
        usage_data = vars(usage)
    else:
        usage_data = {}
    allowed_usage = {
        key: value
        for key, value in usage_data.items()
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
    }
    return ImageAnalysisResponse(
        text=text,
        model=model,
        usage_metadata=allowed_usage or None,
    )
