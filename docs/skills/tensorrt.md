# TensorRT

The TensorRT skill covers NVIDIA TensorRT engine building and deployment for maximum inference performance on NVIDIA GPUs.

**Skill directory:** `skills/tensorrt/`

## Purpose

When ONNX Runtime's CUDA provider isn't fast enough, TensorRT extracts maximum performance from NVIDIA GPUs through layer fusion, kernel auto-tuning, and precision calibration. This skill covers the full pipeline from ONNX model to deployed TensorRT engine, including dynamic shapes, INT8 calibration, and benchmarking.

**Prerequisite:** The ONNX skill. Always export and slim with ONNX before converting to TensorRT.

## When to Use

- Deploying to known NVIDIA GPU hardware where latency is critical
- FP16 or INT8 precision is acceptable for your use case
- You need 2-6x speedup beyond ONNX Runtime CUDA
- Deploying in NVIDIA containers (NGC, Triton Inference Server)

## Key Patterns

### trtexec CLI Conversion

```bash
# ONNX → TensorRT with FP16
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x640x640
```

### ONNX Runtime TensorRT EP (Recommended)

```python
import onnxruntime as ort

providers = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache/",
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

session = ort.InferenceSession("model.onnx", providers=providers)
```

### Full Pipeline

```
PyTorch → torch.onnx.export() → onnxslim.slim() → trtexec/Builder → .engine
```

## Anti-Patterns to Avoid

- Do not skip ONNX export — TensorRT does not consume PyTorch models directly
- Do not skip `onnxslim.slim()` before TensorRT — redundant ops confuse the optimizer
- Do not assume engines are portable — they are tied to the GPU architecture and TensorRT version
- Do not use INT8 without calibration data — accuracy will degrade significantly
- Do not use TensorRT for CPU or non-NVIDIA deployment — use ONNX Runtime instead

## Combines Well With

- **ONNX** -- prerequisite; provides the export and slimming pipeline
- **Docker CV** -- TensorRT containers use NVIDIA NGC base images
- **GCP** -- Vertex AI supports TensorRT-optimized inference containers
- **PyTorch Lightning** -- training produces checkpoints for the export pipeline

## Full Reference

See [`skills/tensorrt/SKILL.md`](https://github.com/ortizeg/ai-cv-claude-skills/blob/main/skills/tensorrt/SKILL.md) for Python Builder API, dynamic shape profiles, INT8 calibration, benchmarking utilities, Pydantic configuration, and Dockerfile patterns.
