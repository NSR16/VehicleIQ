"""Centralized, lazy configuration for VehicleIQ."""

from dataclasses import dataclass, field
from functools import lru_cache
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

from app.errors import ConfigurationError


GEMINI_MODEL = "gemini-3.5-flash-lite"
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True)
class Settings:
    """Runtime settings needed by the Gemini integration."""

    gemini_api_key: str = field(repr=False)
    gemini_model: str = GEMINI_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load the root environment file on first use and validate configuration."""
    load_dotenv(ROOT_ENV_FILE, override=False)
    api_key = getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is not configured. Set it in the root .env or environment."
        )
    return Settings(gemini_api_key=api_key)
