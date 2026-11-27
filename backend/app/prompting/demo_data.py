"""Small, curated research demonstrations packaged with the API."""

from pathlib import Path
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.errors import ConfigurationError
from app.prompting.examples import ContrastivePair, FewShotExample, examples_from_pairs
from app.reference import IndicatorReference

DEMO_IMAGE_DIR = Path(__file__).parent / "demo_images"


def _source(manufacturer: str, record_id: str) -> str:
    return (
        f"Vehicle-icon dataset record {record_id} ({manufacturer}); its exact source "
        "manual, model, and year were not preserved in the record. Provisional "
        "research annotation; do not treat as vehicle-specific verified guidance."
    )


def load_demo_pairs(
    catalogue: dict[str, IndicatorReference],
) -> tuple[ContrastivePair, ...]:
    """Load and validate the four bundled images and catalogue-backed demo labels."""
    entries = (
        ("mopar_rear_cargo_door_open", "249038ec80c400a0", "Mopar", "rear_cargo_door_open"),
        ("mopar_hood_open", "8412b8cc80c400a0", "Mopar", "hood_open"),
        ("volvo_brake_system_warning", "00b2d49696ccf000", "Volvo", "brake_system"),
        ("volvo_abs_warning", "00f2ce9ab2cef000", "Volvo", "abs"),
    )
    examples: dict[str, FewShotExample] = {}
    for example_id, record_id, manufacturer, label_id in entries:
        path = DEMO_IMAGE_DIR / f"{record_id}.png"
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            raise ConfigurationError(
                f"Required demonstration image {record_id}.png is unavailable at {path}."
            ) from exc
        ref = catalogue.get(label_id)
        if ref is None:
            raise ConfigurationError(
                f"Demonstration {example_id} uses missing catalogue label {label_id}."
            )
        examples[example_id] = FewShotExample(
            image_bytes=image_bytes,
            mime_type="image/png",
            label=label_id,
            meaning=ref.meaning,
            example_id=example_id,
            source_note=_source(manufacturer, record_id),
        )
    return (
        ContrastivePair(
            first=examples["mopar_rear_cargo_door_open"],
            second=examples["mopar_hood_open"],
            visual_distinction="The open-part marker is at the rear cargo area versus the front hood area of the side-profile car.",
            semantic_difference="The first indicates a rear cargo door/trunk not closed; the second indicates the hood not closed.",
        ),
        ContrastivePair(
            first=examples["volvo_brake_system_warning"],
            second=examples["volvo_abs_warning"],
            visual_distinction="Both show an amber circled symbol between curved brackets; the center is an exclamation mark versus the letters ABS.",
            semantic_difference="The first is a general brake-system warning; the second specifically indicates an anti-lock braking system warning.",
        ),
    )


def validate_demo_files() -> None:
    """Fail at application startup if any required bundled image is missing."""
    expected = (
        "249038ec80c400a0.png",
        "8412b8cc80c400a0.png",
        "00b2d49696ccf000.png",
        "00f2ce9ab2cef000.png",
    )
    missing = [name for name in expected if not (DEMO_IMAGE_DIR / name).is_file()]
    if missing:
        raise ConfigurationError(
            f"Required demonstration image files are missing from {DEMO_IMAGE_DIR}: {', '.join(missing)}"
        )
    try:
        for name in expected:
            payload = (DEMO_IMAGE_DIR / name).read_bytes()
            with Image.open(BytesIO(payload)) as image:
                if image.format != "PNG":
                    raise ValueError(f"{name} is not a PNG image.")
                image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ConfigurationError(f"Invalid required demonstration image: {exc}") from exc


def demo_example_ids(pairs: tuple[ContrastivePair, ...]) -> list[str]:
    return [example.example_id for example in examples_from_pairs(list(pairs)) if example.example_id]
