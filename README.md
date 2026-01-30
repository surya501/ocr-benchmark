# OCR Benchmark Suite

Compare OCR models on historical US postage stamps.

## Setup

```bash
git clone https://github.com/surya501/ocr-benchmark
cd ocr-benchmark

# Optional: add HuggingFace token for faster downloads
cp .env.sample .env
# Edit .env with your token from https://huggingface.co/settings/tokens
```

## Usage

```bash
just test         # Test LightOnOCR
just test-hunyuan # Test HunyuanOCR
```

Both run with warmup on images in `postage_stamps/` and display results with timing.

## Models

- **LightOnOCR-2-1B** (2.7B params) - Fast, clean text output
- **HunyuanOCR** (1B params) - Multilingual, includes text coordinates

## Architecture

- `benchmark_lightonocr.py` - LightOnOCR with standard transformers
- `benchmark_hunyuan.py` - HunyuanOCR with specific transformers version
- Separate dependencies per model (no conflicts)
- Test images: 3 historical US stamps in `postage_stamps/`

## Adding Models

1. Create `benchmark_[model].py` with load/infer/test functions
2. Add to justfile:
   ```just
   test-mymodel:
       uv run benchmark_mymodel.py --warmup
   ```

## Troubleshooting

**Import errors**: `rm -rf ~/.cache/uv/environments-v2/benchmark-*` and retry

**Out of memory**: Use `--cpu` flag (slower but works)

**Models not found**: Set `HF_TOKEN` in `.env` or add `local_files_only=False` to scripts

## Resources

- [LightOnOCR](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
