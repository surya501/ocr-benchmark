#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-generativeai>=0.3.0",
#     "pillow",
#     "python-dotenv",
# ]
# ///
"""
Gemini OCR benchmarking and testing.

Usage:
    uv run benchmark_gemini.py [--mode MODE] [--runs N] [--folder PATH] [--warmup]

Modes: test, bench (default: test)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai


def load_model():
    """Load Gemini model and configure API."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    print("Gemini-2.5-flash loaded")
    return model


def infer(model, image_path: Path) -> Tuple[str, float]:
    """Run Gemini OCR inference."""
    try:
        image = Image.open(image_path).convert("RGB")

        t0 = time.perf_counter()
        response = model.generate_content([
            "Recognize all text in this image. List each text element found.",
            image
        ])
        elapsed = time.perf_counter() - t0

        result = response.text if response.text else "(no text detected)"
        return result, elapsed
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            raise RuntimeError("Quota exceeded. Free tier limit reached. Please upgrade your API key or wait for quota reset.") from e
        raise


def benchmark(args):
    """Benchmark Gemini model."""
    model = load_model()

    image_path = Path("postage_stamps/smithsonian_01.jpg")
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Timed runs
    times = []
    for i in range(args.runs):
        try:
            _, elapsed = infer(model, image_path)
            times.append(elapsed)
            print(f"Run {i + 1}: {elapsed:.2f}s")
        except RuntimeError as e:
            print(f"Error: {e}")
            return

    print(f"\nAvg: {sum(times) / len(times):.2f}s | Min: {min(times):.2f}s | Max: {max(times):.2f}s")


def test(args):
    """Test Gemini on a folder of images."""
    model = load_model()

    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in args.folder.iterdir() if f.suffix.lower() in extensions])

    if not images:
        print(f"No images found in {args.folder}")
        return

    print(f"Found {len(images)} images in {args.folder}\n")

    # Process all images
    results = []
    times = []

    for img_path in images:
        print(f"Processing: {img_path.name}...", end=" ", flush=True)
        try:
            result, elapsed = infer(model, img_path)
            times.append(elapsed)
            results.append((img_path.name, result, elapsed))
            print(f"{elapsed:.2f}s")
        except RuntimeError as e:
            print(f"\nError: {e}")
            return

    # Output results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, result, elapsed in results:
        print(f"\n[{name}] ({elapsed:.2f}s)")
        print("-" * 40)
        print(result if result else "(empty)")

    # Benchmark summary
    print("\n" + "=" * 70)
    print("BENCHMARK")
    print("=" * 70)
    print(f"Images:      {len(images)}")
    print(f"Total time:  {sum(times):.2f}s")
    print(f"Avg time:    {sum(times) / len(times):.2f}s")
    print(f"Min time:    {min(times):.2f}s")
    print(f"Max time:    {max(times):.2f}s")
    print(f"Throughput:  {len(images) / sum(times):.2f} img/s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Gemini OCR Benchmarking Tool")
    parser.add_argument(
        "--mode", choices=["test", "bench"], default="test", help="Mode: test or benchmark"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--folder", type=Path, default=Path("postage_stamps"), help="Folder with images for testing")
    args = parser.parse_args()

    if args.mode == "bench":
        benchmark(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
