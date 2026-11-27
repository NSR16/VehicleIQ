"""Focused mock-only tests for the research evaluation runner."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app.errors import GeminiServiceError
from app.research.evaluation_runner import (
    STRATEGIES,
    PreflightError,
    RequestStartPacer,
    build_schedule,
    load_manifest,
    prediction_is_correct,
    resolve_and_validate_images,
    run_experiment,
)
from app.reference import load_catalogue
from app.schemas.analysis import Strategy


FIELDS = [
    "original_candidate_order", "original_image_id", "filename", "dataset_relative_path",
    "manufacturer", "human_confirmed_catalogue_id", "human_review_decision",
    "evaluation_status", "sha256",
]


def png_bytes(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def prediction(label_id: str = "abs") -> str:
    return json.dumps({
        "status": "identified", "label_id": label_id, "confidence": "high",
        "visible_colour": "amber", "state": "steady", "visual_evidence": "test shape",
        "clarification_questions": [],
    })


class EvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.dataset = self.base / "dataset"
        self.image_rel = Path("data/groundtruth/icons/test.png")
        self.image_path = self.dataset / self.image_rel
        self.image_path.parent.mkdir(parents=True)
        self.image_bytes = png_bytes()
        self.image_path.write_bytes(self.image_bytes)
        self.manifest = self.base / "manifest.csv"
        self.output = self.base / "predictions.jsonl"
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, **updates):
        row = {
            "original_candidate_order": "7", "original_image_id": "test-record",
            "filename": "test.png", "dataset_relative_path": self.image_rel.as_posix(),
            "manufacturer": "Test Motors", "human_confirmed_catalogue_id": "abs",
            "human_review_decision": "approved", "evaluation_status": "approved_for_evaluation",
            "sha256": hashlib.sha256(self.image_bytes).hexdigest(),
        }
        row.update(updates)
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return row

    def fake_identify(self, response=None, error=None, calls=None):
        def identify(*args, **kwargs):
            if calls is not None:
                calls.append((args, kwargs))
            if error:
                raise error
            return SimpleNamespace(text=response or prediction(), usage_metadata={"total_token_count": 10})
        return identify

    def run_runner(self, **kwargs):
        kwargs.setdefault("sleep", lambda _: None)
        return run_experiment(
            manifest_path=self.manifest,
            dataset_root=self.dataset,
            output_path=self.output,
            delay=0,
            expected_records=1,
            expected_classes=1,
            minimum_start_interval=0,
            **kwargs,
        )

    def test_manifest_loading_and_catalogue_id_validation(self):
        rows = load_manifest(self.manifest, load_catalogue())
        self.assertEqual(rows[0]["original_image_id"], "test-record")
        self.write_manifest(human_confirmed_catalogue_id="not_a_catalogue_id")
        with self.assertRaises(PreflightError):
            load_manifest(self.manifest, load_catalogue())

    def test_runtime_exclusion_filters_without_modifying_manifest(self):
        first = self.write_manifest()
        second = dict(first, original_candidate_order="8", original_image_id="excluded-record")
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([first, second])
        before = self.manifest.read_bytes()
        rows = load_manifest(self.manifest, load_catalogue(), exclude_record_ids=("excluded-record",))
        self.assertEqual([row["original_image_id"] for row in rows], ["test-record"])
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_unknown_runtime_exclusion_is_rejected(self):
        with self.assertRaises(PreflightError):
            load_manifest(self.manifest, load_catalogue(), exclude_record_ids=("not-in-manifest",))

    def test_dataset_path_resolution_hash_and_invalid_image_detection(self):
        rows = load_manifest(self.manifest, load_catalogue())
        records = resolve_and_validate_images(rows, self.dataset)
        self.assertEqual(records[0].image_bytes, self.image_bytes)
        self.image_path.write_bytes(b"not a PNG")
        with self.assertRaises(PreflightError):
            resolve_and_validate_images(rows, self.dataset)

    def test_dataset_path_must_remain_under_dataset_root(self):
        rows = load_manifest(self.manifest, load_catalogue())
        rows[0]["dataset_relative_path"] = "../outside.png"
        with self.assertRaises(PreflightError):
            resolve_and_validate_images(rows, self.dataset)

    def test_schedule_is_paired_in_order(self):
        records = resolve_and_validate_images(load_manifest(self.manifest, load_catalogue()), self.dataset)
        schedule = build_schedule(records)
        self.assertEqual([strategy for _, strategy in schedule], list(STRATEGIES))
        self.assertEqual([record.record_id for record, _ in schedule], ["test-record", "test-record"])
        self.assertEqual(STRATEGIES, (Strategy.ordinary, Strategy.contrastive))

    def test_dry_run_validates_and_makes_zero_identification_calls(self):
        calls = []
        with redirect_stdout(io.StringIO()) as stdout:
            result = self.run_runner(dry_run=True, identify=self.fake_identify(calls=calls))
        self.assertEqual(result["base_requests"], 2)
        self.assertEqual(len(calls), 0)
        self.assertIn("Base requests: 2", stdout.getvalue())
        self.assertFalse(self.output.exists())

    def test_successful_run_records_correctness_then_resume_skips_both_pairs(self):
        calls = []
        identify = self.fake_identify(calls=calls)
        first = self.run_runner(client=object(), identify=identify)
        self.assertEqual(first["completed"], 2)
        self.assertEqual(len(calls), 2)
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(row["correct"] is True for row in rows))
        self.assertEqual(rows[0]["parsed_prediction"]["label_id"], "abs")
        self.assertTrue(all(row["request_status"] == "success" for row in rows))
        self.assertEqual(rows[0]["token_usage"]["total_token_count"], 10)
        second_calls = []
        second = self.run_runner(client=object(), identify=self.fake_identify(calls=second_calls))
        self.assertEqual(second["pending"], 0)
        self.assertEqual(second_calls, [])

    def test_invalid_structured_output_keeps_raw_response_and_error(self):
        self.run_runner(client=object(), identify=self.fake_identify(response="not-json"))
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["request_status"], "parse_error")
        self.assertEqual(rows[0]["raw_model_output"], "not-json")
        self.assertTrue(rows[0]["parse_validation_error"])

    def test_quota_failure_is_recorded_and_stops_scheduling(self):
        calls = []
        self.run_runner(client=object(), identify=self.fake_identify(error=GeminiServiceError(
            "quota", status_code=429, provider_status="RESOURCE_EXHAUSTED",
            provider_message="Daily quota exhausted",
        ), calls=calls))
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_status"], "quota_exhausted")
        self.assertEqual(rows[0]["http_status_code"], 429)
        self.assertEqual(len(calls), 1)

    def test_temporary_rpm_429_retries_with_retry_after_and_does_not_complete_early(self):
        calls = []
        delays = []
        def identify(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise GeminiServiceError(
                    "throttled", status_code=429, provider_status="RESOURCE_EXHAUSTED",
                    provider_message="Requests per minute rate limit exceeded",
                    retry_after_seconds=8,
                )
            return SimpleNamespace(text=prediction(), usage_metadata=None)
        result = self.run_runner(
            client=object(), identify=identify, sleep=delays.append,
            random_uniform=lambda low, high: 0,
        )
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["request_status"] for row in rows], ["rate_limited_retry", "success", "success"])
        self.assertEqual(rows[0]["failure_kind"], "temporary_rpm_throttling")
        self.assertEqual(rows[0]["retry_after_seconds"], 8)
        self.assertEqual(rows[0]["retry_delay_seconds"], 8)
        self.assertEqual(delays[0], 8)
        self.assertEqual(result["retried_requests"], 1)
        self.assertEqual(result["retry_attempts"], 1)
        self.assertEqual(result["completed"], 2)

    def test_start_pacer_accounts_for_request_duration(self):
        now = [0.0]
        waits = []
        def sleep(seconds):
            waits.append(seconds)
            now[0] += seconds
        pacer = RequestStartPacer(6.0, sleep, lambda: now[0])
        pacer.wait_for_start()
        now[0] += 3.0  # first request duration
        pacer.wait_for_start()
        self.assertEqual(waits, [3.0])
        self.assertEqual(pacer.last_start, 6.0)
        now[0] += 8.0  # a slow request needs no additional wait
        pacer.wait_for_start()
        self.assertEqual(waits, [3.0])
        self.assertEqual(pacer.last_start, 14.0)

    def test_transient_failure_retries_once_and_succeeds(self):
        calls = []
        def identify(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise GeminiServiceError("temporary", status_code=503)
            return SimpleNamespace(text=prediction(), usage_metadata=None)
        self.run_runner(client=object(), identify=identify)
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["request_status"] for r in rows], ["transient_retry", "success", "success"])
        self.assertEqual(len(calls), 3)

    def test_api_key_is_redacted_from_raw_model_output_and_logs(self):
        key = "test-api-key-do-not-log"
        output = prediction().replace("test shape", key)
        calls = []
        with patch.dict("os.environ", {"GEMINI_API_KEY": key}), redirect_stdout(io.StringIO()) as stdout:
            self.run_runner(client=object(), identify=self.fake_identify(response=output, calls=calls))
        persisted = self.output.read_text(encoding="utf-8")
        self.assertNotIn(key, persisted)
        self.assertNotIn(key, stdout.getvalue())
        self.assertIn("[REDACTED]", persisted)

    def test_protocol_count_mismatch_blocks_real_provider_calls(self):
        calls = []
        with self.assertRaises(PreflightError):
            run_experiment(
                manifest_path=self.manifest,
                dataset_root=self.dataset,
                output_path=self.output,
                delay=0,
                expected_records=23,
                expected_classes=1,
                client=object(),
                identify=self.fake_identify(calls=calls),
                sleep=lambda _: None,
            )
        self.assertEqual(calls, [])

    def test_result_correctness_uses_exact_catalogue_id(self):
        self.assertTrue(prediction_is_correct("abs", "abs"))
        self.assertFalse(prediction_is_correct("brake_system", "abs"))
        self.assertFalse(prediction_is_correct(None, "abs"))


if __name__ == "__main__":
    unittest.main()
