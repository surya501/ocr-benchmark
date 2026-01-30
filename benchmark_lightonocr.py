#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.5,<2.7",
#     "torchvision",
#     "transformers>=4.45.0",
#     "tokenizers>=0.20",
#     "pillow",
#     "pypdfium2",
#     "requests",
#     "matplotlib",
# ]
# ///
"""
LightOnOCR benchmarking and testing.

Usage:
    uv run benchmark_lightonocr.py [--mode MODE] [--cpu] [--runs N] [--folder PATH] [--warmup]

Modes: test, bench (default: test)
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


def load_model(use_cpu: bool = False):
    """Load LightOnOCR model and processor."""
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"Device: {device}, dtype: {dtype}")
    print("Loading LightOnOCR-2-1B...")
    t0 = time.perf_counter()

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        "lightonai/LightOnOCR-2-1B", torch_dtype=dtype, local_files_only=True
    ).to(device)
    processor = LightOnOcrProcessor.from_pretrained(
        "lightonai/LightOnOCR-2-1B", local_files_only=True
    )

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s\n")

    return model, processor, device, dtype


def infer(model, processor, device, dtype, image_path: Path) -> Tuple[str, float]:
    """Run LightOnOCR inference."""
    image = Image.open(image_path).convert("RGB")
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    result = processor.decode(generated_ids, skip_special_tokens=True)
    return result, elapsed


def benchmark(args):
    """Benchmark LightOnOCR model."""
    model, processor, device, dtype = load_model(args.cpu)

    image_path = Path("postage_stamps/smithsonian_01.jpg")
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Warmup
    print("Warmup run...")
    _, _ = infer(model, processor, device, dtype, image_path)
    print("Warmup done.\n")

    # Timed runs
    times = []
    for i in range(args.runs):
        _, elapsed = infer(model, processor, device, dtype, image_path)
        times.append(elapsed)
        print(f"Run {i + 1}: {elapsed:.2f}s")

    print(f"\nAvg: {sum(times) / len(times):.2f}s | Min: {min(times):.2f}s | Max: {max(times):.2f}s")


def test(args):
    """Test LightOnOCR on a folder of images."""
    model, processor, device, dtype = load_model(args.cpu)

    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in args.folder.iterdir() if f.suffix.lower() in extensions])

    if not images:
        print(f"No images found in {args.folder}")
        return

    print(f"Found {len(images)} images in {args.folder}\n")

    # Warmup
    if args.warmup and images:
        print("Warmup run...")
        _, _ = infer(model, processor, device, dtype, images[0])
        print("Warmup done.\n")

    # Process all images
    results = []
    times = []

    for img_path in images:
        print(f"Processing: {img_path.name}...", end=" ", flush=True)
        result, elapsed = infer(model, processor, device, dtype, img_path)
        times.append(elapsed)
        results.append((img_path.name, result, elapsed))
        print(f"{elapsed:.2f}s")

    # Output results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, result, elapsed in results:
        print(f"\n[{name}] ({elapsed:.2f}s)")
        print("-" * 40)
        print(result.strip() if result.strip() else "(empty)")

    # Benchmark summary
    print("\n" + "=" * 70)
    print("BENCHMARK")
    print("=" * 70)
    print(f"Device:      {device}")
    print(f"Images:      {len(images)}")
    print(f"Total time:  {sum(times):.2f}s")
    print(f"Avg time:    {sum(times) / len(times):.2f}s")
    print(f"Min time:    {min(times):.2f}s")
    print(f"Max time:    {max(times):.2f}s")
    print(f"Throughput:  {len(images) / sum(times):.2f} img/s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LightOnOCR Benchmarking Tool")
    parser.add_argument(
        "--mode", choices=["test", "bench"], default="test", help="Mode: test or benchmark"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU only")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--folder", type=Path, default=Path("postage_stamps"), help="Folder with images for testing")
    parser.add_argument("--warmup", action="store_true", help="Run warmup inference first")
    args = parser.parse_args()

    if args.mode == "bench":
        benchmark(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
