"""Small set of errors exposed by the backend AI modules."""


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


class InvalidImageError(ValueError):
    """Raised when image input does not meet the service contract."""


class GeminiServiceError(RuntimeError):
    """Raised when Gemini cannot provide a usable analysis response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_status: str | None = None,
        provider_message: str | None = None,
        provider_details: object | None = None,
        retry_after_seconds: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_status = provider_status
        self.provider_message = provider_message
        self.provider_details = provider_details
        self.retry_after_seconds = retry_after_seconds
        self.retryable = retryable
