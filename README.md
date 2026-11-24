# VehicleIQ

VehicleIQ is an initial project skeleton for **An Intelligent Vehicle Warning Diagnosis System Using Contrastive Few-Shot Prompting**. The repository contains a React + TypeScript + Vite frontend and a minimal FastAPI backend. AI integration and persistence are reserved for later work.

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

The health endpoint is available at <http://127.0.0.1:8000/health> and returns `{"status":"ok"}`.

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
