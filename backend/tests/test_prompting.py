"""Synthetic-data tests for prompt construction and shared-service integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types
from pydantic import ValidationError

from app.errors import InvalidImageError
from app.prompting.examples import (
    ContrastivePair,
    FewShotExample,
    examples_from_pairs,
)
from app.prompting.prompt_builder import (
    COMMON_TASK,
    build_contrastive_few_shot_content,
    build_ordinary_few_shot_content,
)
from app.prompting.strategies import (
    analyze_contrastive_few_shot,
    analyze_ordinary_few_shot,
)
from app.schemas.analysis import ImageAnalysisResponse, VehicleContext


def example(image: bytes, label: str, meaning: str) -> FewShotExample:
    return FewShotExample(
        image_bytes=image,
        mime_type="image/png",
        label=label,
        meaning=meaning,
    )


def vehicle() -> VehicleContext:
    return VehicleContext(
        manufacturer="Example Motors",
        model="Demo",
        manufacturing_year=2022,
        symptoms="A synthetic test symptom",
    )


def image_parts(content: list[str | types.Part]) -> list[types.Part]:
    return [part for part in content if isinstance(part, types.Part)]


class ExampleRepresentationTests(unittest.TestCase):
    def test_accepts_independent_example_and_pair(self) -> None:
        first = example(b"synthetic-one", "Label A", "Meaning A")
        second = example(b"synthetic-two", "Label B", "Meaning B")
        pair = ContrastivePair(
            first=first,
            second=second,
            visual_distinction="Synthetic shape distinction supplied by the test.",
            semantic_difference="Synthetic meanings differ, as supplied by the test.",
        )
        self.assertEqual(examples_from_pairs([pair]), [first, second])

    def test_rejects_missing_or_invalid_example_data(self) -> None:
        with self.assertRaises(ValidationError):
            FewShotExample(image_bytes=b"image", mime_type="image/png", label="label")
        with self.assertRaises(ValidationError):
            FewShotExample(
                image_bytes=b"image", mime_type="image/gif", label="label", meaning="meaning"
            )
        with self.assertRaises((ValidationError, InvalidImageError)):
            FewShotExample(
                image_bytes=b"", mime_type="image/png", label="label", meaning="meaning"
            )
        with self.assertRaises(ValidationError):
            ContrastivePair(
                first=example(b"one", "A", "A meaning"),
                second=example(b"two", "B", "B meaning"),
                visual_distinction=" ",
                semantic_difference="A supplied difference",
            )


class PromptConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = example(b"synthetic-image-one", "Synthetic label one", "Meaning one")
        self.second = example(b"synthetic-image-two", "Synthetic label two", "Meaning two")
        self.pair = ContrastivePair(
            first=self.first,
            second=self.second,
            visual_distinction="A test-provided difference in outline.",
            semantic_difference="The test examples have different supplied meanings.",
        )
        self.query_bytes = b"synthetic-query-image"
        self.vehicle = vehicle()

    def test_ordinary_prompt_has_independent_examples_and_query(self) -> None:
        content = build_ordinary_few_shot_content(
            [self.first, self.second], self.query_bytes, "image/jpeg", self.vehicle
        )
        text = "\n".join(part for part in content if isinstance(part, str))
        images = image_parts(content)

        self.assertEqual(len(images), 3)
        self.assertEqual([part.inline_data.data for part in images], [
            b"synthetic-image-one", b"synthetic-image-two", self.query_bytes
        ])
        self.assertIn("Synthetic label one", text)
        self.assertIn("Meaning two", text)
        self.assertIn("Example Motors", text)
        self.assertIn("A synthetic test symptom", text)
        self.assertNotIn("visually similar", text.lower())
        self.assertNotIn("visual distinction", text.lower())

    def test_contrastive_prompt_has_both_pair_images_and_supplied_notes(self) -> None:
        content = build_contrastive_few_shot_content(
            [self.pair], self.query_bytes, "image/png", self.vehicle
        )
        text = "\n".join(part for part in content if isinstance(part, str))
        images = image_parts(content)

        self.assertEqual(len(images), 3)
        self.assertEqual([part.inline_data.data for part in images], [
            b"synthetic-image-one", b"synthetic-image-two", self.query_bytes
        ])
        self.assertIn("Synthetic label one", text)
        self.assertIn("Synthetic label two", text)
        self.assertIn(self.pair.visual_distinction, text)
        self.assertIn(self.pair.semantic_difference, text)
        self.assertIn("Example Motors", text)

    def test_strategies_share_context_output_safety_and_same_example_images(self) -> None:
        ordinary = build_ordinary_few_shot_content(
            examples_from_pairs([self.pair]), self.query_bytes, "image/png", self.vehicle
        )
        contrastive = build_contrastive_few_shot_content(
            [self.pair], self.query_bytes, "image/png", self.vehicle
        )
        ordinary_text = ordinary[0]
        contrastive_text = contrastive[0]
        self.assertIsInstance(ordinary_text, str)
        self.assertIsInstance(contrastive_text, str)

        ordinary_common = ordinary_text.split("\n\n", 1)[1]
        contrastive_common = contrastive_text.split("\n\n", 1)[1]
        self.assertEqual(ordinary_common, contrastive_common)
        self.assertEqual(ordinary_common, COMMON_TASK.format(
            manufacturer=self.vehicle.manufacturer,
            model=self.vehicle.model,
            year=self.vehicle.manufacturing_year,
            symptoms=self.vehicle.symptoms,
        ))
        self.assertEqual(
            [part.inline_data.data for part in image_parts(ordinary)],
            [part.inline_data.data for part in image_parts(contrastive)],
        )

    def test_prompt_builders_validate_query_image_and_require_examples(self) -> None:
        with self.assertRaises(InvalidImageError):
            build_ordinary_few_shot_content(
                [self.first], b"", "image/png", self.vehicle
            )
        with self.assertRaises(InvalidImageError):
            build_contrastive_few_shot_content(
                [self.pair], self.query_bytes, "image/gif", self.vehicle
            )
        with self.assertRaises(ValueError):
            build_ordinary_few_shot_content([], self.query_bytes, "image/png", self.vehicle)
        with self.assertRaises(ValueError):
            build_contrastive_few_shot_content([], self.query_bytes, "image/png", self.vehicle)


class PromptingServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = example(b"synthetic-example", "Supplied label", "Supplied meaning")
        self.pair = ContrastivePair(
            first=self.example,
            second=example(b"synthetic-second", "Other label", "Other meaning"),
            visual_distinction="Synthetic supplied distinction.",
            semantic_difference="Synthetic supplied meaning difference.",
        )
        self.client = Mock()
        self.client.models.generate_content.return_value = SimpleNamespace(text="Mock analysis")

    def test_ordinary_strategy_calls_shared_service_with_injected_fake(self) -> None:
        result = analyze_ordinary_few_shot(
            [self.example], b"synthetic-query", "image/png", vehicle(), client=self.client
        )
        self.assertEqual(result, ImageAnalysisResponse(text="Mock analysis", model="gemini-3.5-flash-lite"))
        self.client.models.generate_content.assert_called_once()

    def test_contrastive_strategy_calls_shared_service_with_injected_fake(self) -> None:
        result = analyze_contrastive_few_shot(
            [self.pair], b"synthetic-query", "image/png", vehicle(), client=self.client
        )
        self.assertEqual(result.text, "Mock analysis")
        self.client.models.generate_content.assert_called_once()
        contents = self.client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(sum(isinstance(part, types.Part) for part in contents), 3)


if __name__ == "__main__":
    unittest.main()
