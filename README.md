# VehicleIQ

VehicleIQ is a college research project for **An Intelligent Vehicle Warning Diagnosis System Using Contrastive Few-Shot Prompting**. It contains a React + TypeScript + Vite frontend and a FastAPI backend.

## Requirements

- Node.js and npm
- Python 3.10 or newer

## Run the backend

From the repository root, create and activate a virtual environment, then install the backend dependencies:

```powershell
py -m venv backend/.venv
backend/.venv/Scripts/Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --reload --app-dir backend
```

The health endpoint is available at <http://127.0.0.1:8000/health> and returns `{"status":"ok"}`. The diagnosis endpoint is `POST /api/v1/diagnose` and accepts multipart form data:

| Field | Required | Format |
| --- | --- | --- |
| `image` | Yes | PNG, JPEG, or WebP, up to 10 MB |
| `manufacturer`, `model` | No | Text context |
| `year` | No | Integer from 1886 to 2100 |
| `question` | No | Question about a dashboard indicator |
| `strategy` | No | `ordinary` or `contrastive`; default `ordinary` |
| `answers` | No | JSON object of clarification IDs to text answers |

Set `VEHICLEIQ_DEFAULT_STRATEGY=ordinary` or `contrastive` to configure the default. A successful response includes a catalogue-backed `indicator`; abstentions return `indicator: null`. Each response also includes `status`, `image_status`, a safety disclaimer, and strategy/model/prompt/latency metadata. Uploaded image bytes are processed in memory and are not persisted.

The backend performs an image suitability assessment before indicator identification. The two stages request JSON from Gemini and validate it with Pydantic. Indicator meanings and recommended actions are taken from the local validated catalogue at `backend/app/reference_data/indicators.json`.

The few-shot dataset is not populated yet. Until actual demonstrations are configured, the API explicitly reports `prompt_mode: "zero_shot"` and `few_shot_examples_used: false`; selecting `ordinary` or `contrastive` is recorded as requested metadata and does not currently change the identification prompt.

## Run the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite prints the local development URL when it starts.

## Environment files

`.env.example` files list placeholder variable names only. Copy one to a local `.env` only when needed and add your own values; `.env` files are ignored by Git. The existing root `.env` is local and is not part of this skeleton.
