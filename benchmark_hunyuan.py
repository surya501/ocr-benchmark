#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.5,<2.7",
#     "torchvision",
#     "transformers @ git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4",
#     "tokenizers>=0.20",
#     "pillow",
#     "pypdfium2",
#     "requests",
#     "matplotlib",
#     "accelerate",
# ]
# ///
"""
HunyuanOCR benchmarking and testing.

Usage:
    uv run benchmark_hunyuan.py [--mode MODE] [--cpu] [--runs N] [--folder PATH] [--warmup]

Modes: test, bench (default: test)
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


def load_model(use_cpu: bool = False):
    """Load HunyuanOCR model and processor."""
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Loading HunyuanOCR...")
    t0 = time.perf_counter()

    processor = AutoProcessor.from_pretrained(
        "tencent/HunyuanOCR", use_fast=False
    )
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        "tencent/HunyuanOCR",
        attn_implementation="eager",
        dtype=torch.bfloat16 if not use_cpu else torch.float32,
        device_map="auto" if not use_cpu else "cpu",
    )
    model = model.eval()

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s\n")

    return model, processor, device


def format_ocr_result(raw_result: str) -> str:
    """Format HunyuanOCR output by removing prompt and formatting coordinates nicely."""
    # Remove the prompt prefix
    prompt = "检测并识别图片中的文字。"
    if raw_result.startswith(prompt):
        raw_result = raw_result[len(prompt):]

    # Parse and format text with coordinates
    import re
    # Match pattern: TEXT(x1,y1),(x2,y2)
    pattern = r'([^()]+)\((\d+,\d+)\),\((\d+,\d+)\)'
    matches = re.findall(pattern, raw_result)

    if not matches:
        return raw_result

    formatted_lines = []
    for text, coord1, coord2 in matches:
        formatted_lines.append(f"{text.strip()} [{coord1} - {coord2}]")

    return "\n".join(formatted_lines) if formatted_lines else raw_result


def infer(model, processor, device, image_path: Path) -> Tuple[str, float]:
    """Run HunyuanOCR inference."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "检测并识别图片中的文字。"},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=image, padding=True, return_tensors="pt")

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    inputs = inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    raw_result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    result = format_ocr_result(raw_result)
    return result, elapsed


def benchmark(args):
    """Benchmark HunyuanOCR model."""
    model, processor, device = load_model(args.cpu)

    image_path = Path("postage_stamps/smithsonian_01.jpg")
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Warmup
    print("Warmup run...")
    _, _ = infer(model, processor, device, image_path)
    print("Warmup done.\n")

    # Timed runs
    times = []
    for i in range(args.runs):
        _, elapsed = infer(model, processor, device, image_path)
        times.append(elapsed)
        print(f"Run {i + 1}: {elapsed:.2f}s")

    print(f"\nAvg: {sum(times) / len(times):.2f}s | Min: {min(times):.2f}s | Max: {max(times):.2f}s")


def test(args):
    """Test HunyuanOCR on a folder of images."""
    model, processor, device = load_model(args.cpu)

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
        _, _ = infer(model, processor, device, images[0])
        print("Warmup done.\n")

    # Process all images
    results = []
    times = []

    for img_path in images:
        print(f"Processing: {img_path.name}...", end=" ", flush=True)
        result, elapsed = infer(model, processor, device, img_path)
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
        print(result if result else "(empty)")

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
    parser = argparse.ArgumentParser(description="HunyuanOCR Benchmarking Tool")
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
