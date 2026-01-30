# OCR Benchmark Suite

Comprehensive benchmarking and testing suite for comparing OCR models on historical US postage stamps.

## Models

- **LightOnOCR-2-1B** - Fast, efficient OCR model. Good for clean, modern documents.
- **HunyuanOCR** - Multilingual OCR with text coordinate detection. Better for complex layouts.

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- `uv` package manager

### Setup

1. Clone the repo
```bash
git clone https://github.com/surya501/ocr-benchmark
cd ocr-benchmark
```

2. Configure HuggingFace token (optional but recommended for faster downloads)
```bash
cp .env.sample .env
# Edit .env and add your HF_TOKEN from https://huggingface.co/settings/tokens
```

3. Download models (first run will auto-download, but you can pre-cache):
```bash
# For LightOnOCR (uses standard transformers)
uv run benchmark_lightonocr.py --warmup

# For HunyuanOCR (uses specific transformers version)
uv run benchmark_hunyuan.py --warmup
```

## Usage

### Test on Postage Stamps

```bash
# Test LightOnOCR with warmup
just test

# Test HunyuanOCR with warmup
just test-hunyuan
```

Both commands:
- Run warmup inference first
- Process all images in `postage_stamps/`
- Show OCR results with timing
- Display throughput metrics

### Manual Invocation

```bash
# Test with custom folder
uv run benchmark_lightonocr.py --folder /path/to/images

# Test without warmup
uv run benchmark_lightonocr.py --warmup=false

# Show help
uv run benchmark_lightonocr.py --help
```

## Architecture

### File Structure
- `benchmark_lightonocr.py` - LightOnOCR test script (standard transformers)
- `benchmark_hunyuan.py` - HunyuanOCR test script (specific transformers commit)
- `justfile` - Command shortcuts for common tasks
- `postage_stamps/` - Test images (3 historical US stamps)
- `.env` - HuggingFace token (not committed, use .env.sample as template)

### Dependency Isolation

Each model has **isolated dependencies** to avoid conflicts:
- **LightOnOCR**: Uses `transformers>=4.45.0`
- **HunyuanOCR**: Uses specific transformers commit for compatibility

This means you can test both models without dependency conflicts.

## Output Format

### LightOnOCR
Clean text output:
```
1846 - IOWA STATEHOOD CENTENNIAL - 1946
UNITED STATES POSTAGE 3¢
```

### HunyuanOCR
Text with bounding box coordinates:
```
TEXT [x1,y1 - x2,y2]
TEXT [x1,y1 - x2,y2]
```

## Model Comparison

| Feature | LightOnOCR | HunyuanOCR |
|---------|-----------|-----------|
| Speed | ⭐⭐⭐ Fast | ⭐⭐ Slower |
| Size | 2.7B | 1B |
| Coordinates | ❌ No | ✅ Yes |
| Multilingual | ❌ Limited | ✅ Yes |
| Modern stamps | ✅ Better | ⭐ Ok |
| Historical stamps | ⭐ Ok | ✅ Better |

## Development

### Adding a New Model

1. Create `benchmark_[modelname].py` with your own dependencies
2. Implement `load_model()`, `infer()`, `test()`, and `benchmark()` functions
3. Add to `justfile`:
```just
test-mymodel:
    uv run benchmark_mymodel.py --warmup
```

### Environment

```bash
# View available commands
just

# Check Python version
uv python --version

# Clear environments (if needed)
rm -rf ~/.cache/uv/environments-v2/benchmark-*
```

## Troubleshooting

### Models not found locally
The first run will download models from HuggingFace Hub. If offline:
- Pre-cache models on an online machine
- Set `HF_TOKEN` for faster downloads
- Use `local_files_only=False` in scripts temporarily

### Out of memory
If running on low-VRAM GPU:
- Use `--cpu` flag to test on CPU (slower but works)
- Reduce batch sizes in scripts if needed
- Test one image at a time

### Import errors
If you see `ImportError` for transformers classes:
- Clear cache: `rm -rf ~/.cache/uv/environments-v2/benchmark-*`
- Re-run the command to rebuild environment

## Performance Notes

Typical speeds on NVIDIA GPU:
- **LightOnOCR**: 0.8-1.1s per image
- **HunyuanOCR**: 1.1-4.1s per image (varies by image complexity)

## Contributing

- Add new test images to `postage_stamps/`
- Test both models on your images
- Document any findings in issues
- Keep scripts modular and self-contained

## License

MIT License - See repo for details

## Resources

- [LightOnOCR](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
