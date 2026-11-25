"""Small set of errors exposed by the backend AI modules."""


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


class InvalidImageError(ValueError):
    """Raised when image input does not meet the service contract."""


class GeminiServiceError(RuntimeError):
    """Raised when Gemini cannot provide a usable analysis response."""
