
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load the existing VehicleIQ root .env
load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("ERROR: GEMINI_API_KEY not found in .env")

MODEL = "gemini-3.5-flash-lite"

try:
    client = genai.Client(api_key=api_key)

    print(f"Testing Gemini model: {MODEL}")

    response = client.models.generate_content(
        model=MODEL,
        contents="Reply with exactly: VehicleIQ API connection successful!"
    )

    print("\nSUCCESS!")
    print("Gemini response:")
    print(response.text)

except Exception as e:
    print("\nAPI REQUEST FAILED")
    print(f"Error: {e}")
