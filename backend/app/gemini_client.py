"""Lazy, cached construction of the Google GenAI client."""

from functools import lru_cache

from google import genai
from google.genai import types

from app.config import get_settings
from app.errors import GeminiServiceError


REQUEST_TIMEOUT_MS = 60_000


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Create one client on first use, without retrying failed requests."""
    settings = get_settings()
    try:
        return genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    except Exception as exc:
        raise GeminiServiceError("Could not initialize the Gemini client.") from exc
