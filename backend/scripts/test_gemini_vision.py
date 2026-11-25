"""Manually test Gemini image input with one local dashboard warning image."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


MODEL = "gemini-3.5-flash-lite"
PROMPT = (
    "Identify the dashboard warning symbol shown in this image. Give its name and "
    "briefly explain what it commonly indicates. If uncertain, say so."
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class InvalidImageError(ValueError):
    """Raised when the image is missing, malformed, or unsupported."""


def fail(message: str, exit_code: int = 2) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def detect_image_format(data: bytes) -> str:
    """Identify supported image formats from their file signatures."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise InvalidImageError("The PNG file is truncated or malformed.")
        width, height = struct.unpack(">II", data[16:24])
        if width == 0 or height == 0:
            raise InvalidImageError("The PNG image has invalid dimensions.")
        return "image/png"

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx"}:
            return "image/heic"
        if brand in {b"mif1", b"msf1"}:
            return "image/heif"

    raise InvalidImageError(
        "Unsupported or unrecognized image format. Supported formats: "
        "PNG, JPEG, WEBP, HEIC, and HEIF."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one local dashboard warning image to Gemini for identification."
    )
    parser.add_argument("image", type=Path, help="Path to one image file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the result JSON (default: backend/scripts/output/...) ",
    )
    return parser.parse_args()


def write_result(output_path: Path, result: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        fail(f"Image file does not exist: {image_path}")

    try:
        image_bytes = image_path.read_bytes()
        mime_type = detect_image_format(image_bytes)
    except (OSError, InvalidImageError) as exc:
        fail(f"Invalid image: {exc}")

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        fail(f"GEMINI_API_KEY was not found in the environment or {repo_root / '.env'}")

    timestamp = datetime.now(timezone.utc).isoformat()
    output_path = args.output.expanduser().resolve() if args.output else (
        DEFAULT_OUTPUT_DIR
        / f"gemini_vision_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}.json"
    )
    metadata: dict[str, object] = {
        "timestamp_utc": timestamp,
        "model": MODEL,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_mime_type": mime_type,
        "image_size_bytes": len(image_bytes),
        "prompt": PROMPT,
    }

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=MODEL,
            contents=[PROMPT, image_part],
        )
        response_text = response.text or ""
        write_result(output_path, {**metadata, "response": response_text})
        print("Gemini response:")
        print(response_text if response_text else "(No text returned.)")
        print(f"\nSaved response and metadata to: {output_path}")
        return 0
    except errors.APIError as exc:
        code = getattr(exc, "code", None)
        if code == 429:
            summary = "Gemini quota or rate limit reached; no retry was attempted."
        else:
            summary = f"Gemini API request failed (HTTP {code})."
        result = {**metadata, "error": summary, "error_details": str(exc)}
        try:
            write_result(output_path, result)
            print(f"Saved failure metadata to: {output_path}", file=sys.stderr)
        except OSError as write_exc:
            print(f"Could not save failure metadata: {write_exc}", file=sys.stderr)
        print(f"ERROR: {summary}\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        summary = f"Gemini request failed: {type(exc).__name__}: {exc}"
        try:
            write_result(output_path, {**metadata, "error": summary})
            print(f"Saved failure metadata to: {output_path}", file=sys.stderr)
        except OSError as write_exc:
            print(f"Could not save failure metadata: {write_exc}", file=sys.stderr)
        print(f"ERROR: {summary}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
