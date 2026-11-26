"""Entry points that build strategy-specific content and call the shared AI service."""

from collections.abc import Sequence

from app.prompting.examples import ContrastivePair, FewShotExample, ImageMimeType
from app.prompting.prompt_builder import (
    build_contrastive_few_shot_content,
    build_ordinary_few_shot_content,
)
from app.schemas.analysis import ImageAnalysisResponse, VehicleContext
from app.services.ai_service import GeminiClient, analyze_content


def analyze_ordinary_few_shot(
    examples: Sequence[FewShotExample],
    query_image_bytes: bytes,
    query_mime_type: ImageMimeType,
    vehicle: VehicleContext,
    *,
    client: GeminiClient | None = None,
) -> ImageAnalysisResponse:
    """Run the ordinary few-shot strategy through the shared Gemini service."""
    content = build_ordinary_few_shot_content(
        examples, query_image_bytes, query_mime_type, vehicle
    )
    return analyze_content(content, client=client)


def analyze_contrastive_few_shot(
    pairs: Sequence[ContrastivePair],
    query_image_bytes: bytes,
    query_mime_type: ImageMimeType,
    vehicle: VehicleContext,
    *,
    client: GeminiClient | None = None,
) -> ImageAnalysisResponse:
    """Run contrastive few-shot content through the shared Gemini service."""
    content = build_contrastive_few_shot_content(
        pairs, query_image_bytes, query_mime_type, vehicle
    )
    return analyze_content(content, client=client)
