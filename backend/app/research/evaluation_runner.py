"""Paired, resumable evaluation runner for the approved VehicleIQ image set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from app.api.diagnose import (
    PROMPT_VERSION,
    RESEARCH_QUESTION,
    Identification,
    identify_dashboard_indicator,
)
from app.config import GEMINI_MODEL
from app.errors import ConfigurationError, GeminiServiceError
from app.gemini_client import get_gemini_client
from app.prompting.demo_data import load_demo_pairs, validate_demo_files
from app.reference import load_catalogue
from app.schemas.analysis import Strategy
from app.services.ai_service import MAX_IMAGE_SIZE_BYTES


EXPECTED_APPROVED_RECORDS = 23
EXPECTED_CATALOGUE_CLASSES = 11
STRATEGIES = (Strategy.ordinary, Strategy.contrastive)
TEMPERATURE = 0.0
MAX_TRANSIENT_RETRIES = 2
MAX_RATE_LIMIT_RETRIES = 3
MIN_REQUEST_START_INTERVAL_SECONDS = 6.0
TRANSIENT_HTTP_CODES = {408, 500, 502, 503, 504}
MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "analysis_output" / "vehicleIQ_review" / "final_evaluation_manifest.csv"
DEFAULT_OUTPUT = ROOT / "backend" / "results" / "research_evaluation" / "predictions.jsonl"


class PreflightError(ValueError):
    """Raised when evaluation inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class EvaluationRecord:
    candidate_order: str
    record_id: str
    image_id: str
    filename: str
    manufacturer: str
    ground_truth_catalogue_id: str
    image_path: Path
    image_bytes: bytes
    mime_type: str
    sha256: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise PreflightError(f"Cannot read manifest: {path}") from exc
    if not rows:
        raise PreflightError("The approved manifest is empty or has no data rows.")
    return rows


def load_manifest(
    path: Path,
    catalogue: dict[str, Any],
    *,
    exclude_record_ids: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Load approved rows, applying explicitly requested runtime-only exclusions."""
    rows = _read_csv(path)
    required = {
        "original_candidate_order", "original_image_id", "filename", "dataset_relative_path",
        "manufacturer", "human_confirmed_catalogue_id", "human_review_decision",
        "evaluation_status", "sha256",
    }
    missing_columns = required.difference(rows[0])
    if missing_columns:
        raise PreflightError(f"Manifest is missing required columns: {', '.join(sorted(missing_columns))}")
    seen: set[str] = set()
    excluded = set(exclude_record_ids)
    found_exclusions: set[str] = set()
    for row in rows:
        record_id = row["original_image_id"].strip()
        label_id = row["human_confirmed_catalogue_id"].strip()
        if not record_id or record_id in seen:
            raise PreflightError("Manifest record IDs must be non-empty and unique.")
        seen.add(record_id)
        if row["human_review_decision"].strip().lower() != "approved" or row["evaluation_status"].strip() != "approved_for_evaluation":
            raise PreflightError(f"Record {record_id} is not marked approved for evaluation.")
        if not label_id or label_id not in catalogue:
            raise PreflightError(f"Record {record_id} has an unrecognized confirmed catalogue ID.")
        if len(row["sha256"].strip()) != 64:
            raise PreflightError(f"Record {record_id} has no valid SHA-256 manifest value.")
        if record_id in excluded:
            found_exclusions.add(record_id)
    unknown_exclusions = excluded - found_exclusions
    if unknown_exclusions:
        raise PreflightError(f"Requested exclusion IDs are not in the manifest: {', '.join(sorted(unknown_exclusions))}")
    selected = [row for row in rows if row["original_image_id"].strip() not in excluded]
    if not selected:
        raise PreflightError("Runtime exclusions removed every approved record.")
    return selected


def resolve_and_validate_images(
    rows: Iterable[dict[str, str]], dataset_root: Path
) -> list[EvaluationRecord]:
    """Resolve manifest-relative paths and fully decode/hash every query before calls."""
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise PreflightError(f"Dataset root is not a directory: {root}")
    records: list[EvaluationRecord] = []
    for row in rows:
        record_id = row["original_image_id"].strip()
        relative = Path(row["dataset_relative_path"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise PreflightError(f"Manifest path escapes the dataset root for record {record_id}.")
        if not path.is_file():
            raise PreflightError(f"Image is missing for record {record_id}: {relative}")
        try:
            payload = path.read_bytes()
            if len(payload) > MAX_IMAGE_SIZE_BYTES:
                raise PreflightError(f"Image exceeds the production request size limit for record {record_id}.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image_format = image.format
                    image.load()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, ValueError) as exc:
            raise PreflightError(f"Image cannot be decoded for record {record_id}: {relative}") from exc
        mime = MIME_BY_FORMAT.get(image_format or "")
        if mime is None:
            raise PreflightError(f"Unsupported image format for record {record_id}: {image_format}")
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash.lower() != row["sha256"].strip().lower():
            raise PreflightError(f"Image SHA-256 does not match the approved manifest for record {record_id}.")
        if row["filename"].strip() != path.name:
            raise PreflightError(f"Filename/path mismatch for record {record_id}.")
        records.append(EvaluationRecord(
            candidate_order=row["original_candidate_order"].strip(),
            record_id=record_id,
            image_id=record_id,
            filename=path.name,
            manufacturer=row["manufacturer"].strip(),
            ground_truth_catalogue_id=row["human_confirmed_catalogue_id"].strip(),
            image_path=path,
            image_bytes=payload,
            mime_type=mime,
            sha256=actual_hash,
        ))
    return records


def build_schedule(records: Iterable[EvaluationRecord]) -> list[tuple[EvaluationRecord, Strategy]]:
    """Create paired observations in source order, ordinary then contrastive."""
    return [(record, strategy) for record in records for strategy in STRATEGIES]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _redact(value: str | None) -> str | None:
    if value is None:
        return None
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return value.replace(key, "[REDACTED]") if key else value


def _redact_object(value: Any) -> Any:
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact_object(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_object(item) for key, item in value.items()}
    return value


def prediction_is_correct(predicted_id: str | None, ground_truth_id: str) -> bool:
    """Treat abstention or a different ID as incorrect for exact-class accuracy."""
    return predicted_id == ground_truth_id


def _load_success_keys(path: Path, run_config_id: str) -> set[tuple[str, str, str]]:
    successful: set[tuple[str, str, str]] = set()
    if not path.exists():
        return successful
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(f"Invalid JSONL result at line {line_number}; refusing to resume.") from exc
            if not isinstance(item, dict):
                raise PreflightError(f"Invalid JSONL result at line {line_number}; refusing to resume.")
            if (item.get("request_status") == "success" and item.get("run_config_id") == run_config_id):
                successful.add((item.get("record_id", ""), item.get("strategy", ""), run_config_id))
    return successful


def _load_run_rows(path: Path, run_config_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(f"Invalid JSONL result at line {line_number}; refusing to resume.") from exc
            if not isinstance(item, dict):
                raise PreflightError(f"Invalid JSONL result at line {line_number}; refusing to resume.")
            if item.get("run_config_id") == run_config_id:
                rows.append(item)
    return rows


def _request_counts(path: Path, run_config_id: str, scheduled: int) -> dict[str, int]:
    rows = _load_run_rows(path, run_config_id)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    retry_statuses = {"rate_limited_retry", "transient_retry"}
    retry_rows = [row for row in rows if row.get("request_status") in retry_statuses]
    for row in rows:
        latest[(str(row.get("record_id", "")), str(row.get("strategy", "")))] = row
    completed = sum(row.get("request_status") == "success" for row in latest.values())
    terminal_failures = {
        "parse_error", "provider_error", "quota_exhausted", "rate_limit_retries_exhausted",
        "transient_retries_exhausted",
    }
    failed = sum(row.get("request_status") in terminal_failures for row in latest.values())
    pending = max(0, scheduled - completed - failed)
    retried_requests = len({
        (str(row.get("record_id", "")), str(row.get("strategy", ""))) for row in retry_rows
    })
    return {
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "retried_requests": retried_requests,
        "retry_attempts": len(retry_rows),
    }


def _is_quota_exhaustion(exc: GeminiServiceError) -> bool:
    parts = [exc.provider_status or "", exc.provider_message or ""]
    if exc.provider_details is not None:
        try:
            parts.append(json.dumps(exc.provider_details, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            parts.append(str(exc.provider_details))
    details = " ".join(parts).lower().replace("-", "_").replace(" ", "_")
    rpm_markers = ("per_minute", "perminute", "requests_per_minute", "rate_limit", "rate_limited")
    if any(marker in details for marker in rpm_markers):
        return False
    quota_markers = (
        "per_day", "perday", "daily_quota", "daily_limit", "billing", "quota_exceeded",
        "quota_exhausted", "exceeded_your_current_quota", "quota_has_been_exhausted",
    )
    if any(marker in details for marker in quota_markers):
        return True
    return exc.status_code == 403 and ("resource_exhausted" in details or "quota" in details)


def _retry_delay_seconds(
    attempt: int,
    *,
    retry_after_seconds: float | None,
    random_uniform: Callable[[float, float], float],
) -> float:
    exponential_with_jitter = (2 ** (attempt - 1)) * 6.0 + random_uniform(0.0, 2.0)
    return max(retry_after_seconds or 0.0, exponential_with_jitter)


def _rate_limit_kind(exc: GeminiServiceError) -> str:
    parts = [exc.provider_status or "", exc.provider_message or ""]
    if exc.provider_details is not None:
        try:
            parts.append(json.dumps(exc.provider_details, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            parts.append(str(exc.provider_details))
    details = " ".join(parts).lower().replace("-", "_").replace(" ", "_")
    if any(marker in details for marker in ("per_minute", "perminute", "requests_per_minute", "rate_limit", "rate_limited", "rpm")):
        return "temporary_rpm_throttling"
    return "temporary_429_unclassified"


class RequestStartPacer:
    """Enforce a minimum interval between request starts, not request completions."""

    def __init__(self, interval_seconds: float, sleep: Callable[[float], None], clock: Callable[[], float]):
        self.interval_seconds = interval_seconds
        self.sleep = sleep
        self.clock = clock
        self.last_start: float | None = None

    def wait_for_start(self) -> None:
        now = self.clock()
        if self.last_start is not None:
            wait_seconds = max(0.0, self.interval_seconds - (now - self.last_start))
            if wait_seconds:
                self.sleep(wait_seconds)
        self.last_start = self.clock()


def _append_result(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    return {key: value for key, value in usage.items() if isinstance(value, int) and not isinstance(value, bool)}


def _make_run_config(
    records: list[EvaluationRecord], manifest_path: Path, pairs: tuple[Any, ...], delay: float,
    effective_interval: float, excluded_record_ids: tuple[str, ...],
) -> dict[str, Any]:
    demo_metadata = []
    for pair in pairs:
        for example in (pair.first, pair.second):
            demo_metadata.append({
                "example_id": example.example_id,
                "sha256": hashlib.sha256(example.image_bytes).hexdigest(),
                "mime_type": example.mime_type,
                "label_id": example.label,
                "meaning": example.meaning,
            })
    response_schema = Identification.model_json_schema()
    contrastive_pairs = [
        {
            "first_example_id": pair.first.example_id,
            "second_example_id": pair.second.example_id,
            "visual_distinction": pair.visual_distinction,
            "semantic_difference": pair.semantic_difference,
        }
        for pair in pairs
    ]
    settings = {
        "model": GEMINI_MODEL,
        "prompt_version": PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "response_schema_sha256": _json_hash(response_schema),
        "question": RESEARCH_QUESTION,
        "strategies": [strategy.value for strategy in STRATEGIES],
        "demonstrations_in_order": demo_metadata,
        "contrastive_pair_notes": contrastive_pairs,
        "manifest_sha256": _sha256_file(manifest_path),
        "query_records": [
            {"record_id": r.record_id, "image_sha256": r.sha256, "manufacturer": r.manufacturer,
             "ground_truth_catalogue_id": r.ground_truth_catalogue_id}
            for r in records
        ],
        "requested_delay_seconds": delay,
        "effective_minimum_request_start_interval_seconds": effective_interval,
        "maximum_target_requests_per_minute": 60.0 / effective_interval if effective_interval else None,
        "max_transient_retries": MAX_TRANSIENT_RETRIES,
        "max_rate_limit_retries": MAX_RATE_LIMIT_RETRIES,
        "backoff_base_seconds": 6,
        "backoff_jitter_max_seconds": 2,
        "retry_after_header": "Retry-After",
        "expected_protocol_records": EXPECTED_APPROVED_RECORDS,
        "expected_catalogue_classes": EXPECTED_CATALOGUE_CLASSES,
        "runtime_excluded_record_ids": list(excluded_record_ids),
    }
    return {"run_config_id": _json_hash(settings), "created_at": _now(), "settings": settings}


def _persist_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_config_id") != config["run_config_id"]:
            raise PreflightError(f"Run configuration file already exists with different settings: {path}")


def _write_summary(path: Path, values: dict[str, Any]) -> None:
    lines = ["# VehicleIQ research evaluation run", ""]
    for key, value in values.items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
    delay: float = 4.0,
    dry_run: bool = False,
    expected_records: int = EXPECTED_APPROVED_RECORDS,
    expected_classes: int = EXPECTED_CATALOGUE_CLASSES,
    exclude_record_ids: Iterable[str] = (),
    minimum_start_interval: float = MIN_REQUEST_START_INTERVAL_SECONDS,
    client: Any | None = None,
    identify: Callable[..., Any] = identify_dashboard_indicator,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> dict[str, Any]:
    """Validate every input before any provider request; append one record per attempt."""
    if delay < 0 or minimum_start_interval < 0:
        raise PreflightError("Delay and minimum request-start interval must be non-negative.")
    effective_interval = max(delay, minimum_start_interval)
    exclusions = tuple(sorted(set(exclude_record_ids)))
    manifest_path = manifest_path.expanduser().resolve()
    rows = load_manifest(manifest_path, load_catalogue(), exclude_record_ids=exclusions)
    records = resolve_and_validate_images(rows, dataset_root)
    validate_demo_files()
    pairs = load_demo_pairs(load_catalogue())
    schedule = build_schedule(records)
    class_count = len({record.ground_truth_catalogue_id for record in records})
    count_matches_protocol = len(records) == expected_records and class_count == expected_classes
    config = _make_run_config(records, manifest_path, pairs, delay, effective_interval, exclusions)
    config_path = output_path.with_name(f"{output_path.stem}.{config['run_config_id'][:12]}.config.json")
    summary_path = output_path.with_name(f"{output_path.stem}.{config['run_config_id'][:12]}.summary.md")
    _persist_config(config_path, config)

    successful = _load_success_keys(output_path, config["run_config_id"])
    pending = [
        (record, strategy) for record, strategy in schedule
        if (record.record_id, strategy.value, config["run_config_id"]) not in successful
    ]
    expected_calls = len(records) * len(STRATEGIES)
    print(f"Approved manifest records: {len(records)}")
    print(f"Catalogue classes: {class_count}")
    if exclusions:
        print(f"Runtime exclusions: {', '.join(exclusions)}")
    print(f"Paired strategies: {len(STRATEGIES)} ({', '.join(s.value for s in STRATEGIES)})")
    print(f"Base requests: {expected_calls}; already successful: {len(successful)}; remaining: {len(pending)}")
    print(f"Minimum interval between request starts: {effective_interval:.2f}s (maximum {60/effective_interval if effective_interval else 'unlimited'} requests/minute)")
    print(f"Run config: {config_path}")
    print(f"Prediction log: {output_path}")
    if not count_matches_protocol:
        print(f"PROTOCOL COUNT MISMATCH: expected {expected_records} records across {expected_classes} classes ({expected_records*2} calls), selected input contains {len(records)} records across {class_count} classes ({expected_calls} calls). Real requests are blocked.")
    if dry_run:
        _write_summary(summary_path, {
            "status": "preflight only; no Gemini calls made",
            "approved_manifest_records": len(records),
            "catalogue_classes": class_count,
            "runtime_excluded_record_ids": list(exclusions),
            "expected_protocol_records": expected_records,
            "expected_catalogue_classes": expected_classes,
            "base_requests": expected_calls,
            "remaining_requests_for_this_run_config": len(pending),
            "protocol_count_matches": count_matches_protocol,
            "requested_delay_seconds": delay,
            "effective_minimum_request_start_interval_seconds": effective_interval,
            "maximum_target_requests_per_minute": 60.0 / effective_interval if effective_interval else None,
            "prediction_log": output_path,
            "run_config": config_path,
        })
        print(f"Summary: {summary_path}")
        return {"record_count": len(records), "base_requests": expected_calls, "pending_requests": len(pending), "protocol_count_matches": count_matches_protocol, "dry_run": True}
    if not count_matches_protocol:
        raise PreflightError("No Gemini requests sent because the approved manifest count does not match the evaluation protocol.")

    active_client = client if client is not None else get_gemini_client()
    catalogue = load_catalogue()
    quota_exhausted = False
    pacer = RequestStartPacer(effective_interval, sleep, monotonic)
    for record, strategy in pending:
        attempt = 0
        rate_limit_retries = 0
        transient_retries = 0
        while True:
            attempt += 1
            pacer.wait_for_start()
            started = time.perf_counter()
            context = json.dumps({
                "manufacturer": record.manufacturer,
                "model": None,
                "year": None,
                "question": RESEARCH_QUESTION,
                "clarification_answers": {},
            }, ensure_ascii=False)
            raw_output: str | None = None
            prediction: Identification | None = None
            parse_error: str | None = None
            usage: dict[str, int] | None = None
            request_status = "success"
            status_code: int | None = None
            failure_kind: str | None = None
            retry_after_seconds: float | None = None
            retry_delay_seconds: float | None = None
            should_retry = False
            try:
                analysis = identify(
                    record.image_bytes, record.mime_type, context, strategy,
                    catalogue, pairs, client=active_client,
                )
                raw_output = _redact(analysis.text)
                usage = _usage_dict(analysis.usage_metadata)
                prediction = Identification.model_validate_json(analysis.text)
            except ValidationError as exc:
                request_status = "parse_error"
                parse_error = _redact(str(exc))
            except GeminiServiceError as exc:
                status_code = exc.status_code
                retry_after_seconds = exc.retry_after_seconds
                if _is_quota_exhaustion(exc):
                    request_status = "quota_exhausted"
                    failure_kind = "daily_or_billing_quota_exhaustion"
                    quota_exhausted = True
                elif status_code == 429:
                    failure_kind = _rate_limit_kind(exc)
                    if rate_limit_retries < MAX_RATE_LIMIT_RETRIES:
                        request_status = "rate_limited_retry"
                        rate_limit_retries += 1
                        should_retry = True
                        retry_delay_seconds = _retry_delay_seconds(
                            rate_limit_retries,
                            retry_after_seconds=retry_after_seconds,
                            random_uniform=random_uniform,
                        )
                    else:
                        request_status = "rate_limit_retries_exhausted"
                elif (status_code in TRANSIENT_HTTP_CODES or exc.retryable) and transient_retries < MAX_TRANSIENT_RETRIES:
                    request_status = "transient_retry"
                    failure_kind = "transient_http_or_network_error"
                    transient_retries += 1
                    should_retry = True
                    retry_delay_seconds = _retry_delay_seconds(
                        transient_retries,
                        retry_after_seconds=retry_after_seconds,
                        random_uniform=random_uniform,
                    )
                elif status_code in TRANSIENT_HTTP_CODES or exc.retryable:
                    request_status = "transient_retries_exhausted"
                    failure_kind = "transient_http_or_network_retries_exhausted"
                else:
                    request_status = "provider_error"
                    failure_kind = "non_retryable_or_retries_exhausted"
            except (ValueError, TypeError) as exc:
                request_status = "parse_error"
                parse_error = _redact(str(exc))
            except Exception:
                # Do not serialize provider exception text; it may contain credentials or request data.
                request_status = "provider_error"

            predicted_id = prediction.label_id if prediction is not None and prediction.status == "identified" else None
            latency_ms = max(0, int((time.perf_counter() - started) * 1000))
            row = {
                "timestamp_utc": _now(),
                "original_candidate_order": record.candidate_order,
                "record_id": record.record_id,
                "image_id": record.image_id,
                "filename": record.filename,
                "manufacturer": record.manufacturer,
                "ground_truth_catalogue_id": record.ground_truth_catalogue_id,
                "strategy": strategy.value,
                "run_config_id": config["run_config_id"],
                "attempt": attempt,
                "predicted_catalogue_id": predicted_id,
                "prediction_status": prediction.status if prediction else None,
                "parsed_prediction": _redact_object(prediction.model_dump(mode="json")) if prediction else None,
                "correct": prediction_is_correct(predicted_id, record.ground_truth_catalogue_id),
                "raw_model_output": raw_output,
                "parse_validation_error": parse_error,
                "model_name": GEMINI_MODEL,
                "prompt_version": PROMPT_VERSION,
                "temperature": TEMPERATURE,
                "latency_ms": latency_ms,
                "request_status": request_status,
                "http_status_code": status_code,
                "failure_kind": failure_kind,
                "retry_after_seconds": retry_after_seconds,
                "retry_delay_seconds": retry_delay_seconds,
                "token_usage": usage,
            }
            _append_result(output_path, row)
            if request_status == "success":
                successful.add((record.record_id, strategy.value, config["run_config_id"]))
                break
            if should_retry:
                sleep(retry_delay_seconds or 0.0)
                continue
            if quota_exhausted:
                break
            break
        if quota_exhausted:
            break

    counts = _request_counts(output_path, config["run_config_id"], expected_calls)
    if quota_exhausted:
        run_status = "stopped on daily/billing quota exhaustion"
    elif counts["failed"]:
        run_status = "completed with failed requests"
    elif counts["pending"]:
        run_status = "incomplete with pending requests"
    else:
        run_status = "completed successfully"
    _write_summary(summary_path, {
        "status": run_status,
        "approved_manifest_records": len(records),
        "base_requests": expected_calls,
        **counts,
        "effective_minimum_request_start_interval_seconds": effective_interval,
        "maximum_target_requests_per_minute": 60.0 / effective_interval if effective_interval else None,
        "max_transient_retries": MAX_TRANSIENT_RETRIES,
        "max_rate_limit_retries": MAX_RATE_LIMIT_RETRIES,
        "runtime_excluded_record_ids": list(exclusions),
        "prediction_log": output_path,
        "run_config": config_path,
    })
    print(f"Completed: {counts['completed']}; pending: {counts['pending']}; failed: {counts['failed']}; retried requests: {counts['retried_requests']}; retry attempts: {counts['retry_attempts']}")
    print(f"Summary: {summary_path}")
    return {"record_count": len(records), "base_requests": expected_calls, **counts, "quota_exhausted": quota_exhausted, "dry_run": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=os.getenv("VEHICLEIQ_DATASET_ROOT"), help="Root containing data/groundtruth/icons; may also use VEHICLEIQ_DATASET_ROOT.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=4.0, help="Requested minimum gap between request starts; the runner enforces at least 6 seconds (default: 4).")
    parser.add_argument("--exclude-record-id", action="append", default=[], help="Exclude this approved record from this run only; may be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report calls without invoking Gemini.")
    args = parser.parse_args(argv)
    if args.dataset_root is None:
        parser.error("--dataset-root or VEHICLEIQ_DATASET_ROOT is required")
    try:
        run_experiment(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_path=args.output,
            delay=args.delay,
            dry_run=args.dry_run,
            exclude_record_ids=args.exclude_record_id,
        )
    except (PreflightError, ConfigurationError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
