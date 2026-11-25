"""Mock-only tests for configuration, client setup, and image analysis."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import errors

from app import config, gemini_client
from app.errors import ConfigurationError, GeminiServiceError, InvalidImageError
from app.schemas.analysis import ImageAnalysisResponse
from app.services.ai_service import analyze_image


class ConfigurationTests(unittest.TestCase):
    def tearDown(self) -> None:
        config.get_settings.cache_clear()

    def test_missing_key_raises_configuration_error(self) -> None:
        config.get_settings.cache_clear()
        with patch.dict(os.environ, {}, clear=True), patch.object(config, "load_dotenv"):
            with self.assertRaises(ConfigurationError):
                config.get_settings()

    def test_settings_load_root_env_and_hide_key_from_repr(self) -> None:
        config.get_settings.cache_clear()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-secret"}, clear=True):
            with patch.object(config, "load_dotenv") as load_dotenv:
                settings = config.get_settings()
        load_dotenv.assert_called_once_with(config.ROOT_ENV_FILE, override=False)
        self.assertEqual(settings.gemini_model, config.GEMINI_MODEL)
        self.assertNotIn("mock-secret", repr(settings))


class GeminiClientTests(unittest.TestCase):
    def tearDown(self) -> None:
        gemini_client.get_gemini_client.cache_clear()

    def test_client_is_cached_and_has_timeout_without_retries(self) -> None:
        gemini_client.get_gemini_client.cache_clear()
        settings = SimpleNamespace(gemini_api_key="mock-secret")
        with patch.object(gemini_client, "get_settings", return_value=settings):
            with patch.object(gemini_client.genai, "Client") as client_factory:
                first = gemini_client.get_gemini_client()
                second = gemini_client.get_gemini_client()

        self.assertIs(first, second)
        client_factory.assert_called_once()
        options = client_factory.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, gemini_client.REQUEST_TIMEOUT_MS)
        self.assertEqual(options.retry_options.attempts, 1)


class ImageAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.client.models.generate_content.return_value = SimpleNamespace(
            text="This appears to be the oil pressure warning."
        )

    def test_sends_image_and_prompt_through_injected_client(self) -> None:
        result = analyze_image(
            b"fake-png-bytes",
            "image/png",
            "Identify this symbol.",
            client=self.client,
        )

        self.assertIsInstance(result, ImageAnalysisResponse)
        self.assertEqual(result.text, "This appears to be the oil pressure warning.")
        self.assertEqual(result.model, config.GEMINI_MODEL)
        self.client.models.generate_content.assert_called_once()
        call = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(call["model"], config.GEMINI_MODEL)
        self.assertEqual(call["contents"][0], "Identify this symbol.")
        self.assertEqual(call["contents"][1].inline_data.mime_type, "image/png")
        self.assertEqual(call["contents"][1].inline_data.data, b"fake-png-bytes")

    def test_rejects_invalid_image_input_without_calling_client(self) -> None:
        invalid_inputs = [
            (b"", "image/png"),
            (b"bytes", "image/gif"),
            (b"x" * (10 * 1024 * 1024 + 1), "image/jpeg"),
        ]
        for image_bytes, mime_type in invalid_inputs:
            with self.subTest(mime_type=mime_type, length=len(image_bytes)):
                with self.assertRaises(InvalidImageError):
                    analyze_image(image_bytes, mime_type, client=self.client)
        self.client.models.generate_content.assert_not_called()

    def test_wraps_provider_api_failure_without_exposing_details(self) -> None:
        self.client.models.generate_content.side_effect = errors.APIError(
            code=500,
            response_json={"error": {"message": "private provider detail"}},
        )

        with self.assertRaises(GeminiServiceError) as raised:
            analyze_image(b"fake-bytes", "image/png", client=self.client)

        self.assertNotIn("private provider detail", str(raised.exception))
        self.client.models.generate_content.assert_called_once()

    def test_rejects_empty_model_response(self) -> None:
        self.client.models.generate_content.return_value = SimpleNamespace(text="  ")
        with self.assertRaises(GeminiServiceError):
            analyze_image(b"fake-bytes", "image/webp", client=self.client)


if __name__ == "__main__":
    unittest.main()
