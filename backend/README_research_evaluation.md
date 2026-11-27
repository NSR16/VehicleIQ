# VehicleIQ research evaluation runner

Run these commands from the repository's `backend` directory. The manifest paths
are relative to the original dataset repository root, so `--dataset-root` must
point to the directory that contains `data/groundtruth/icons/`.

```powershell
$datasetRoot = 'C:\Users\shrir\Downloads\generating-vehicular-icon-descriptions-main\generating-vehicular-icon-descriptions-main'
.\.venv\Scripts\python.exe -m app.research.evaluation_runner --dataset-root $datasetRoot --exclude-record-id 02c8d4d6c2d4d802 --dry-run
```

The dry run reads the approved manifest, checks catalogue IDs, verifies the
approved image hashes and decodes every image, validates the four bundled
demonstrations, and reports request counts. It does not initialize a Gemini
client or call the API.

The real experiment command is:

```powershell
.\.venv\Scripts\python.exe -m app.research.evaluation_runner --dataset-root $datasetRoot --exclude-record-id 02c8d4d6c2d4d802 --delay 4
```

Add `--manifest PATH` and/or `--output PATH` to use alternate files. The dataset
root can also be supplied through `VEHICLEIQ_DATASET_ROOT`. `--delay` controls
the requested minimum gap between request starts; the runner enforces at least
six seconds (at most 10 starts per minute), even when `--delay 4` is used. This
start-to-start pacing accounts for request duration. A Gemini `Retry-After`
header is respected. Temporary HTTP 429 throttles retry up to three times with
exponential backoff and jitter. Transient network and 5xx failures retry up to
two times. Daily quota or billing exhaustion stops the run; successful
predictions are appended immediately and skipped on resume.

## Input count check

The experiment protocol expects 23 records across 11 classes and 46 base calls.
When a runtime-only exclusion is required, pass `--exclude-record-id ID`; the
runner records it in the run config and leaves the input manifest unchanged.
The runner checks both record and class counts before any live call.

## Output files

By default, the runner writes:

- `backend/results/research_evaluation/predictions.jsonl`: append-only per-attempt
  results; successful record/strategy/config combinations are skipped on
  resume. Failed attempts remain recorded.
- A run-config JSON and a human-readable summary Markdown file alongside the
  JSONL, both named with a stable run-config hash.

The run config records the model, prompt version, temperature, response schema
hash, question, demo order and hashes, manifest hash, query hashes and
ground-truth IDs, and pacing/retry settings. Each JSONL row records timestamp,
candidate order and ID, image metadata, manufacturer, ground-truth ID, strategy,
parsed prediction, raw model text, exact-ID correctness, validation errors,
model/prompt/settings, latency, request status, HTTP status where available,
and token counts when returned by Gemini. API keys are not written to outputs.

The runner calls `identify_dashboard_indicator` in `app.api.diagnose`, which is
also used by `POST /api/v1/diagnose`. This shared helper builds the existing
identification prompt, calls the existing shared Gemini service with the
production JSON schema and temperature, and uses the same demonstration data.
Only the endpoint's preceding image-suitability request is bypassed.

All tests use mock clients. No live Gemini request is made by the test suite.
