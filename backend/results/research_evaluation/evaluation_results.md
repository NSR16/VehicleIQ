# VehicleIQ Research Evaluation Results

**Status: COMPLETE.** The paired experiment completed for all selected records. The input manifest was not modified; candidate 7 (`02c8d4d6c2d4d802`) was excluded only at runtime as requested.

## Methodology

The evaluation compares ordinary and contrastive few-shot prompting on the same 23 query images. Every image was submitted once per strategy, in record order, yielding 23 paired observations and 46 successful base requests. The run reused VehicleIQ's production identification helper and schema while bypassing the production suitability request. The four demonstration images, labels, order, query context, model, and generation settings were shared across strategies; only the prescribed prompt framing differs.

The query question was: "Identify the dashboard indicator shown in the image." The manufacturer from each approved manifest row was included as context; model, year, and clarification answers were unset. The contrastive pairs retained their existing visual distinction and semantic difference notes.

| Setting | Value |
|---|---|
| Model | `gemini-3.5-flash-lite` |
| Prompt version | `1.0` |
| Temperature | `0.0` |
| Output | JSON validated against the production `Identification` schema |
| Demonstrations in order | mopar_rear_cargo_door_open (`rear_cargo_door_open`), mopar_hood_open (`hood_open`), volvo_brake_system_warning (`brake_system`), volvo_abs_warning (`abs`) |
| Dataset | 23 selected records from `analysis_output/vehicleIQ_review/final_evaluation_manifest.csv` |
| Catalogue classes | 11 |
| Runtime exclusion | Candidate 7, `02c8d4d6c2d4d802` |
| Request pacing | Minimum 6.0s between starts (maximum 10 starts/minute) |
| Retry policy | `Retry-After` respected; temporary 429 retries up to 3 times with exponential backoff and jitter; transient network/5xx retries up to 2 times |

Before calling Gemini, the runner decoded every included image and verified its SHA-256 against the approved manifest. It also validated the four bundled demonstration images. Both strategies were scheduled for every selected record.

## Overall results

| Strategy | Correct | Incorrect | Valid predictions / scheduled | Accuracy |
|---|---:|---:|---:|---:|
| Ordinary | 22 | 1 | 23 / 23 | 95.7% |
| Contrastive | 22 | 1 | 23 / 23 | 95.7% |

**Accuracy difference (contrastive minus ordinary): +0.0 percentage points.** Both strategies returned the same number of correct classifications.

## Per-class accuracy

| Ground-truth catalogue ID | n | Ordinary correct / n (accuracy) | Contrastive correct / n (accuracy) |
|---|---:|---:|---:|
| `abs` | 3 | 3/3 (100.0%) | 3/3 (100.0%) |
| `airbag_srs` | 3 | 3/3 (100.0%) | 3/3 (100.0%) |
| `battery_charging` | 2 | 1/2 (50.0%) | 1/2 (50.0%) |
| `brake_system` | 3 | 3/3 (100.0%) | 3/3 (100.0%) |
| `coolant_temperature` | 2 | 2/2 (100.0%) | 2/2 (100.0%) |
| `hood_open` | 2 | 2/2 (100.0%) | 2/2 (100.0%) |
| `malfunction_indicator` | 3 | 3/3 (100.0%) | 3/3 (100.0%) |
| `oil_pressure` | 1 | 1/1 (100.0%) | 1/1 (100.0%) |
| `rear_cargo_door_open` | 2 | 2/2 (100.0%) | 2/2 (100.0%) |
| `seat_belt` | 1 | 1/1 (100.0%) | 1/1 (100.0%) |
| `tire_pressure` | 1 | 1/1 (100.0%) | 1/1 (100.0%) |
| **Total** | **23** | **22/23 (95.7%)** | **22/23 (95.7%)** |

## Confusion matrix

The two strategies produced **identical confusion matrices**. Rows are confirmed ground-truth catalogue IDs; columns are predicted IDs. `no_prediction` includes valid `unknown` responses with no catalogue ID. Counts are based on all 23 successful structured responses per strategy.

| Ground truth \ Predicted | `abs` | `airbag_srs` | `battery_charging` | `brake_system` | `coolant_temperature` | `hood_open` | `malfunction_indicator` | `no_prediction` | `oil_pressure` | `rear_cargo_door_open` | `seat_belt` | `tire_pressure` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `abs` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `airbag_srs` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `battery_charging` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `brake_system` | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `coolant_temperature` | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hood_open` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `malfunction_indicator` | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| `oil_pressure` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `rear_cargo_door_open` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| `seat_belt` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `tire_pressure` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## Paired outcomes and statistical comparison

| Paired outcome | Count |
|---|---:|
| Both correct | 22 |
| Ordinary only correct | 0 |
| Contrastive only correct | 0 |
| Both incorrect | 1 |
| Incomplete pairs | 0 |

McNemar's exact two-sided test has **0 discordant pairs** (ordinary-only 0, contrastive-only 0); exact p = **1.000**. With no discordant observations, this test provides no evidence that either strategy outperformed the other. The observed accuracy difference is 0.0 percentage points; this small sample does not establish equivalence or rule out an effect on other images.

## Latency and token usage

Latency is the runner's elapsed time around the shared identification helper, including prompt assembly, provider response, and structured parsing. All successful requests are included.

| Strategy | n | Mean latency | Median latency | Requests with token usage |
|---|---:|---:|---:|---:|
| Ordinary | 23 | 1958.2 ms | 1863.0 ms | 23 |
| Contrastive | 23 | 1522.3 ms | 1441.0 ms | 23 |

| Strategy | Prompt tokens total (mean/request) | Candidate tokens total (mean/request) | Total tokens total (mean/request) |
|---|---:|---:|---:|
| Ordinary | 150,568 (6546.4) | 1,958 (85.1) | 152,526 (6631.6) |
| Contrastive | 153,696 (6682.4) | 1,978 (86.0) | 155,674 (6768.4) |

## Completion and retries

| Measure | Count |
|---|---:|
| Completed strategy predictions | 46 / 46 |
| Pending | 0 |
| Failed requests | 0 |
| Requests retried | 0 |
| Retry attempts | 0 |
| Incomplete record pairs | 0 |
| Daily/billing quota stops | 0 |

All 46 requests completed successfully on their first attempt. The runner persisted every prediction immediately. No successful request was retried.

## Interpretation and limitations

The observed result is a tie: ordinary and contrastive prompting each achieved 22/23 (95.7%), and both missed the same record (`0519386468694131`, ground truth `battery_charging`). Both responses abstained with `status: unknown`, describing a plug symbol that was not represented in the supported catalogue. The different confidence levels did not change exact-class correctness.

This result does not show a contrastive-prompting improvement on this evaluation set. The evaluation contains only 23 images across 11 classes, with several classes represented by one or two examples. It is a single run using one model and one fixed demonstration set; the findings should not be generalized as proof of no effect. The candidate-level predictions and source ordering remain available for further analysis.

## Reproducibility

- Run config: `backend/results/research_evaluation/predictions.fce47c4e6235.config.json`
- Append-only raw prediction log: `backend/results/research_evaluation/predictions.jsonl`
- Runner summary: `backend/results/research_evaluation/predictions.fce47c4e6235.summary.md`
- Run config ID: `fce47c4e6235163ab205dcdcb8c03dd4a9f2fe03816ae8185d3593b9ab8d5f11`
- Runtime exclusion: `02c8d4d6c2d4d802`; neither approved manifest nor review CSV was edited
- Effective request-start interval: 6.0 seconds; no retry or quota event occurred

The JSONL contains all 46 raw model responses, parsed structured predictions, exact-class correctness, latency, request status, and token usage metadata. The run config preserves the manifest and image hashes, catalogue IDs, question, schema, model settings, demo image hashes/order, and contrastive notes. The experiment completed using the live Gemini API; all reported metrics above are computed from this run's persisted JSONL.
