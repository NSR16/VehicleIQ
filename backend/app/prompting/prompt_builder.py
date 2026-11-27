"""Build multimodal prompt content for the two few-shot strategies."""

from collections.abc import Sequence

from google.genai import types

from app.errors import InvalidImageError
from app.schemas.analysis import VehicleContext
from app.services.ai_service import MAX_IMAGE_SIZE_BYTES
from app.prompting.examples import ContrastivePair, FewShotExample, ImageMimeType


COMMON_TASK = """Identify the dashboard warning symbol shown in the query image using the supplied demonstrations and vehicle context.

Vehicle context:
Manufacturer: {manufacturer}
Model: {model}
Manufacturing year: {year}
Symptoms: {symptoms}

Respond in readable text with these sections:
1. Likely identification.
2. Visible evidence: describe only visual characteristics actually visible in the query image.
3. General meaning of the symbol.
4. Possible implications: distinguish possibilities from confirmed facts.
5. Recommended action: give reasonable preliminary guidance.
6. Uncertainty: say whether the image is clear, ambiguous, or unsupported; mention plausible alternatives or request a clearer image/additional information when appropriate.

Do not force a definitive identification, invent a confidence percentage, or treat a model-reported percentage as calibrated probability. Separate visible evidence, general symbol meaning, possible causes, and what cannot be determined from the image alone. Do not infer a specific mechanical fault, sensor reading, vehicle specification, or manufacturer instruction unless supplied. You are not a substitute for the vehicle manual or a qualified mechanic. For potentially serious warnings, advise consulting the manual and seeking professional assistance where appropriate."""

ORDINARY_FRAMING = (
    "Use each labeled demonstration as an independent example. Apply the supplied "
    "labels and meanings to the query without adding relationships between demonstrations."
)

CONTRASTIVE_FRAMING = (
    "The demonstrations are grouped into pairs that may look similar but have "
    "different meanings. Attend to the visual distinction and semantic difference "
    "provided for each pair. Use only those supplied distinction notes; do not "
    "invent additional visual characteristics."
)


def _validated_image_part(image_bytes: bytes, mime_type: str) -> types.Part:
    if not image_bytes:
        raise InvalidImageError("Image bytes must not be empty.")
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise InvalidImageError("Image MIME type must be PNG, JPEG, or WebP.")
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise InvalidImageError("Image exceeds the maximum size of 10 MB.")
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def _base_prompt(
    strategy_framing: str, vehicle: VehicleContext, task_prompt: str | None = None
) -> str:
    if task_prompt is not None:
        return f"{strategy_framing}\n\n{task_prompt}"
    symptoms = vehicle.symptoms.strip() if vehicle.symptoms else "Not provided"
    context = COMMON_TASK.format(
        manufacturer=vehicle.manufacturer,
        model=vehicle.model,
        year=vehicle.manufacturing_year,
        symptoms=symptoms,
    )
    return f"{strategy_framing}\n\n{context}"


def _example_text(index: int, example: FewShotExample) -> str:
    example_id = f" (ID: {example.example_id})" if example.example_id else ""
    source = f" Source: {example.source_note}." if example.source_note else ""
    return (
        f"Demonstration {index}{example_id}: label ID: {example.label}; "
        f"meaning: {example.meaning}.{source} Image follows."
    )


def _query_parts(
    parts: list[str | types.Part], query_image_bytes: bytes, query_mime_type: str
) -> list[str | types.Part]:
    parts.extend(
        [
            "Query image follows. Identify this symbol using the instructions above.",
            _validated_image_part(query_image_bytes, query_mime_type),
        ]
    )
    return parts


def build_ordinary_few_shot_content(
    examples: Sequence[FewShotExample],
    query_image_bytes: bytes,
    query_mime_type: ImageMimeType,
    vehicle: VehicleContext | None,
    *,
    task_prompt: str | None = None,
) -> list[str | types.Part]:
    """Construct a request with independent labeled demonstrations."""
    if not examples:
        raise ValueError("At least one few-shot example is required.")
    parts: list[str | types.Part] = [_base_prompt(ORDINARY_FRAMING, vehicle, task_prompt)]
    for index, example in enumerate(examples, start=1):
        parts.extend(
            [
                _example_text(index, example),
                _validated_image_part(example.image_bytes, example.mime_type),
            ]
        )
    return _query_parts(parts, query_image_bytes, query_mime_type)


def build_contrastive_few_shot_content(
    pairs: Sequence[ContrastivePair],
    query_image_bytes: bytes,
    query_mime_type: ImageMimeType,
    vehicle: VehicleContext | None,
    *,
    task_prompt: str | None = None,
) -> list[str | types.Part]:
    """Construct a request with paired examples and supplied distinctions."""
    if not pairs:
        raise ValueError("At least one contrastive pair is required.")
    parts: list[str | types.Part] = [_base_prompt(CONTRASTIVE_FRAMING, vehicle, task_prompt)]
    for pair_number, pair in enumerate(pairs, start=1):
        parts.append(
            f"Pair {pair_number}. Supplied visual distinction: {pair.visual_distinction} "
            f"Supplied semantic difference: {pair.semantic_difference}"
        )
        for side, example in (("A", pair.first), ("B", pair.second)):
            example_id = f" (ID: {example.example_id})" if example.example_id else ""
            source = f" Source: {example.source_note}." if example.source_note else ""
            parts.extend(
                [
                    f"Pair {pair_number}, example {side}{example_id}: label ID: {example.label}; "
                    f"meaning: {example.meaning}.{source} Image follows.",
                    _validated_image_part(example.image_bytes, example.mime_type),
                ]
            )
    return _query_parts(parts, query_image_bytes, query_mime_type)
