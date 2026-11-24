"""FastAPI entry point for VehicleIQ."""

from fastapi import FastAPI

app = FastAPI(title="VehicleIQ API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight service health status."""
    return {"status": "ok"}
