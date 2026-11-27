"""API contract tests use only a fake Gemini client."""
import json
import tempfile
import unittest
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.errors import ConfigurationError
from app.reference import IndicatorReference, load_catalogue

def image_bytes(format_name):
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format=format_name)
    return buffer.getvalue()


PNG = image_bytes("PNG")


class FakeModels:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(text=value)


class DiagnosisApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def post(self, responses, **form):
        fake = SimpleNamespace(models=FakeModels(*responses))
        files = {"image": ("dash.png", PNG, "image/png")}
        with patch("app.api.diagnose.get_gemini_client", return_value=fake):
            result = self.client.post("/api/v1/diagnose", data=form, files=files)
        return result, fake.models

    def test_successful_ordinary_and_contrastive_use_catalogue_fields(self):
        assessment = json.dumps({"image_status": "suitable", "reason": "clear dashboard"})
        identify = json.dumps({"status": "identified", "label_id": "oil_pressure", "confidence": "high", "visible_colour": "red", "state": "steady", "visual_evidence": "red oil-can outline"})
        for strategy in ("ordinary", "contrastive"):
            with self.subTest(strategy=strategy):
                response, fake = self.post([assessment, identify], strategy=strategy)
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["indicator"]["meaning"], load_catalogue()["oil_pressure"].meaning)
                self.assertEqual(body["metadata"]["strategy"], strategy)
                self.assertFalse(body["metadata"]["few_shot_examples_used"])
                self.assertEqual(body["metadata"]["prompt_mode"], "zero_shot")
                self.assertEqual(len(fake.calls), 2)

    def test_contextual_natural_language_questions_are_accepted_and_passed_to_prompt(self):
        assessment = json.dumps({"image_status": "suitable", "reason": "clear dashboard"})
        unknown = json.dumps({"status": "unknown", "label_id": None, "confidence": "low", "state": "unknown", "visual_evidence": "symbol is ambiguous"})
        for question in ("Is this serious?", "What should I do?", "Can I continue driving?"):
            with self.subTest(question=question):
                response, fake = self.post([assessment, unknown], question=question)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "unknown")
                prompt = fake.calls[1]["contents"][0]
                self.assertIn(question, prompt)
                self.assertIn('"question":', prompt)
                self.assertIn("claim the vehicle is safe to drive", prompt)

    def test_ordinary_and_contrastive_have_same_zero_shot_identification_prompt(self):
        assessment = json.dumps({"image_status": "suitable", "reason": "clear"})
        unknown = json.dumps({"status": "unknown", "label_id": None, "confidence": "low", "state": "unknown", "visual_evidence": "ambiguous"})
        prompts = []
        image_parts = []
        for strategy in ("ordinary", "contrastive"):
            response, fake = self.post([assessment, unknown], strategy=strategy, manufacturer="Toyota", question="Is this serious?")
            self.assertEqual(response.status_code, 200)
            prompts.append(fake.calls[1]["contents"][0])
            image_parts.append(fake.calls[1]["contents"][1].inline_data.data)
            self.assertEqual(fake.calls[1]["config"].temperature, 0.0)
            self.assertEqual(fake.calls[1]["config"].response_mime_type, "application/json")
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(image_parts, [PNG, PNG])

    def test_default_strategy_and_invalid_strategy(self):
        with patch.dict("os.environ", {"VEHICLEIQ_DEFAULT_STRATEGY": "contrastive"}):
            response, _ = self.post([json.dumps({"image_status": "irrelevant_image", "reason": "not dashboard"})])
        self.assertEqual(response.json()["metadata"]["strategy"], "contrastive")
        response, _ = self.post([], strategy="experimental")
        self.assertEqual(response.status_code, 422)

    def test_missing_empty_unsupported_and_oversized_images(self):
        missing = self.client.post("/api/v1/diagnose")
        self.assertEqual(missing.status_code, 422)
        for payload, mime, expected in [(b"", "image/png", 400), (b"x", "image/gif", 415), (b"not an image", "image/png", 415), (PNG, "image/jpeg", 415), (b"x" * (10 * 1024 * 1024 + 1), "image/png", 413)]:
            response = self.client.post("/api/v1/diagnose", files={"image": ("x", payload, mime)})
            self.assertEqual(response.status_code, expected)

    def test_jpeg_and_webp_images_are_accepted(self):
        for payload, mime in ((image_bytes("JPEG"), "image/jpeg"), (image_bytes("WEBP"), "image/webp")):
            fake = SimpleNamespace(models=FakeModels(json.dumps({"image_status": "irrelevant_image", "reason": "not a dashboard"})))
            with patch("app.api.diagnose.get_gemini_client", return_value=fake):
                response = self.client.post("/api/v1/diagnose", files={"image": ("image", payload, mime)})
            self.assertEqual(response.status_code, 200)

    def test_all_suitability_abstentions(self):
        for image_status in ("unclear_image", "irrelevant_image", "no_indicator_visible"):
            with self.subTest(image_status=image_status):
                response, fake = self.post([json.dumps({"image_status": image_status, "reason": "test"})])
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["indicator"])
                self.assertEqual(len(fake.calls), 1)

    def test_unknown_and_unrecognized_reference_id_never_return_diagnosis(self):
        suitability = json.dumps({"image_status": "suitable", "reason": "clear"})
        for ident in [
            {"status": "unknown", "label_id": None, "confidence": "low", "state": "unknown", "visual_evidence": "unclear symbol"},
            {"status": "identified", "label_id": "made_up", "confidence": "high", "state": "steady", "visual_evidence": "shape"},
        ]:
            response, _ = self.post([suitability, json.dumps(ident)])
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["indicator"])

    def test_clarification_answers_are_accepted_and_answered_question_not_repeated(self):
        response, fake = self.post([
            json.dumps({"image_status": "suitable", "reason": "clear"}),
            json.dumps({"status": "needs_clarification", "label_id": None, "confidence": "low", "state": "unknown", "visual_evidence": "ambiguous", "clarification_questions": [{"id": "state", "question": "Is it flashing?"}]}),
        ], answers='{"state":"steady"}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unknown")
        self.assertEqual(response.json()["clarification_questions"], [])
        self.assertIn('"state": "steady"', fake.calls[1]["contents"][0])
        bad, _ = self.post([], answers="not-json")
        self.assertEqual(bad.status_code, 422)

    def test_unanswered_clarification_is_returned_once_with_answered_ids_removed(self):
        response, _ = self.post([
            json.dumps({"image_status": "suitable", "reason": "clear"}),
            json.dumps({"status": "needs_clarification", "label_id": None, "confidence": "low", "state": "unknown", "visual_evidence": "ambiguous", "clarification_questions": [{"id": "state", "question": "Is it flashing?"}, {"id": "colour", "question": "What colour is it?"}, {"id": "colour", "question": "What colour is it?"}]}),
        ], answers='{"state":"steady"}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "needs_clarification")
        self.assertEqual([q["id"] for q in response.json()["clarification_questions"]], ["colour"])

    def test_invalid_identification_enum_and_contradictory_status_are_safe_502(self):
        suitability = json.dumps({"image_status": "suitable", "reason": "clear"})
        outputs = [
            {"status": "confident", "label_id": None, "confidence": "high", "state": "steady", "visual_evidence": "shape"},
            {"status": "identified", "label_id": None, "confidence": "high", "state": "steady", "visual_evidence": "shape"},
        ]
        for output in outputs:
            response, _ = self.post([suitability, json.dumps(output)])
            self.assertEqual(response.status_code, 502)

    def test_visible_unsupported_symbol_is_unknown_not_irrelevant(self):
        response, fake = self.post([
            json.dumps({"image_status": "suitable", "reason": "dashboard with a visible unfamiliar symbol"}),
            json.dumps({"status": "unknown", "label_id": None, "confidence": "low", "state": "unknown", "visual_evidence": "visible unfamiliar symbol"}),
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["image_status"], "suitable")
        self.assertEqual(response.json()["status"], "unknown")
        self.assertEqual(len(fake.calls), 2)

    def test_provider_failure_and_malformed_output_are_502(self):
        failed, _ = self.post([RuntimeError("provider")])
        self.assertEqual(failed.status_code, 502)
        malformed, _ = self.post(["not json"])
        self.assertEqual(malformed.status_code, 502)
        with patch("app.api.diagnose.get_gemini_client", side_effect=ConfigurationError("missing key")):
            unavailable = self.client.post("/api/v1/diagnose", files={"image": ("x.png", PNG, "image/png")})
        self.assertEqual(unavailable.status_code, 503)

    def test_health_and_catalogue_validation(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertGreaterEqual(len(load_catalogue()), 10)
        with self.assertRaises(Exception):
            IndicatorReference.model_validate({"label_id": "bad id", "display_name": "x", "category": "n/a", "meaning": "x", "severity": "urgent", "recommended_action": "x", "source_note": "x"})

    def test_catalogue_loader_raises_configuration_error_for_invalid_file(self):
        load_catalogue.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as directory:
                bad_file = Path(directory) / "invalid.json"
                bad_file.write_text("{}", encoding="utf-8")
                with patch("app.reference.DATA_FILE", bad_file):
                    with self.assertRaises(ConfigurationError):
                        load_catalogue()
        finally:
            load_catalogue.cache_clear()


if __name__ == "__main__":
    unittest.main()
