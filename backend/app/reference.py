"""Validated local reference catalogue; fail clearly if packaged data is invalid."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.errors import ConfigurationError

DATA_FILE = Path(__file__).parent / "reference_data" / "indicators.json"


class IndicatorReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=2)
    category: Literal["warning", "status", "informational"]
    meaning: str = Field(min_length=10)
    severity: Literal["high", "medium", "low", "informational"]
    recommended_action: str = Field(min_length=10)
    source_note: str = Field(min_length=10)


@lru_cache(maxsize=1)
def load_catalogue() -> dict[str, IndicatorReference]:
    try:
        rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        items = [IndicatorReference.model_validate(row) for row in rows]
        catalogue = {item.label_id: item for item in items}
        if len(items) < 10 or len(catalogue) != len(items):
            raise ValueError("Catalogue must have at least 10 entries and unique label IDs.")
        return catalogue
    except Exception as exc:
        raise ConfigurationError(f"Invalid indicator reference catalogue at {DATA_FILE}.") from exc
