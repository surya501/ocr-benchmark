# OCR Benchmarking

set dotenv-load := true

# Test LightOnOCR with warmup
test:
    uv run benchmark_lightonocr.py --warmup

# Test HunyuanOCR with warmup
test-hunyuan:
    uv run benchmark_hunyuan.py --warmup

# Test Gemini with warmup
test-gemini:
    uv run benchmark_gemini.py --warmup
