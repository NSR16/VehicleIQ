"""FastAPI entry point for VehicleIQ."""

from fastapi import FastAPI
from app.api.diagnose import router as diagnosis_router
from app.reference import load_catalogue

load_catalogue()
app = FastAPI(title="VehicleIQ API", version="0.1.0")
app.include_router(diagnosis_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight service health status."""
    return {"status": "ok"}
