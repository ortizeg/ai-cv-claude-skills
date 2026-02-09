---
name: tensorrt
description: >
  Maximize inference performance on NVIDIA GPUs by converting ONNX models to TensorRT
  engines. Covers precision modes (FP16/INT8), dynamic shapes, engine building,
  calibration, benchmarking, and Triton Inference Server deployment.
---

# TensorRT Skill

Maximize inference performance on NVIDIA GPUs by converting ONNX models to TensorRT engines. This skill requires the ONNX skill — always export and slim with ONNX first, then convert to TensorRT.

## Prerequisites: ONNX First

TensorRT does not consume PyTorch models directly. The required pipeline is:

```
PyTorch model
  → torch.onnx.export()          (ONNX skill)
  → onnxslim.slim()              (ONNX skill)
  → trtexec / tensorrt.Builder   (this skill)
  → .engine file for deployment
```

Never skip the ONNX export and slimming steps. A slimmed ONNX model produces a better TensorRT engine because redundant ops are already removed before TensorRT's own optimizer runs.

## Why TensorRT

ONNX Runtime with CUDAExecutionProvider is good. TensorRT is better when you need maximum throughput on NVIDIA hardware:

- **2-6x faster** than ONNX Runtime CUDA on the same GPU
- **Kernel auto-tuning** — selects the fastest kernel for each layer on your specific GPU
- **Layer fusion** — combines convolution + batch norm + activation into a single kernel
- **Precision calibration** — FP16 and INT8 inference with minimal accuracy loss
- **Memory optimization** — minimizes GPU memory allocations and data transfers
- **Dynamic shapes** — supports variable batch sizes and image dimensions

The tradeoff: TensorRT engines are GPU-specific (an engine built on A100 will not run on T4) and take minutes to build. Use TensorRT when you deploy to known NVIDIA hardware and need the lowest latency.

## Installation

```bash
# Install TensorRT via pip (requires CUDA toolkit)
pip install tensorrt

# Or via pixi (conda-forge)
# pixi add tensorrt

# Verify installation
python -c "import tensorrt; print(tensorrt.__version__)"
```

TensorRT requires a matching CUDA version. Check compatibility:

| TensorRT | CUDA | cuDNN |
|----------|------|-------|
| 10.x | 12.x | 9.x |
| 8.6 | 11.8 / 12.x | 8.9 |

## Building Engines with trtexec (CLI)

`trtexec` is the fastest way to convert an ONNX model. It ships with TensorRT.

### Basic Conversion

```bash
# Convert ONNX to TensorRT engine (FP32)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine

# FP16 precision (recommended default for GPU inference)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_fp16.engine \
    --fp16

# INT8 precision (requires calibration data)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_int8.engine \
    --int8 \
    --calib=calibration_cache.bin
```

### Dynamic Shapes

```bash
# Variable batch size (1 to 32, optimize for 8)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x640x640

# Variable batch and image size
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --minShapes=input:1x3x320x320 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x1280x1280
```

### Benchmarking with trtexec

```bash
# Benchmark throughput and latency
trtexec \
    --onnx=model.onnx \
    --fp16 \
    --iterations=1000 \
    --warmUp=500 \
    --avgRuns=100

# Output includes:
# - Throughput (inferences/sec)
# - Latency (min, max, mean, median, p99)
# - GPU compute time
# - Host latency (end-to-end)
```

## Building Engines with Python API

### Basic Builder

```python
from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from loguru import logger


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    max_batch_size: int = 1,
    max_workspace_size_gb: int = 4,
) -> None:
    """Build a TensorRT engine from an ONNX model."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    onnx_path = Path(onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TensorRT ONNX parse error: {}", parser.get_error(i))
            msg = f"Failed to parse ONNX model: {onnx_path}"
            raise RuntimeError(msg)

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    # Build engine
    logger.info("Building TensorRT engine from {} (this may take several minutes)", onnx_path.name)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        msg = "Failed to build TensorRT engine"
        raise RuntimeError(msg)

    # Save engine
    engine_path = Path(engine_path)
    engine_path.write_bytes(serialized_engine)
    logger.info("Engine saved to {} ({:.1f} MB)", engine_path, engine_path.stat().st_size / 1e6)
```

### Dynamic Shape Builder

```python
import tensorrt as trt

from loguru import logger


def build_engine_dynamic(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    min_shape: tuple[int, ...] = (1, 3, 320, 320),
    opt_shape: tuple[int, ...] = (8, 3, 640, 640),
    max_shape: tuple[int, ...] = (32, 3, 1280, 1280),
    input_name: str = "input",
) -> None:
    """Build a TensorRT engine with dynamic input shapes."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("{}", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    logger.info(
        "Building engine: min={}, opt={}, max={}",
        min_shape, opt_shape, max_shape,
    )

    serialized = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    logger.info("Engine saved to {}", engine_path)
```

## Inference with TensorRT

### Basic Inference

```python
from __future__ import annotations

import numpy as np
import tensorrt as trt
from cuda import cudart


class TensorRTInference:
    """TensorRT inference session."""

    def __init__(self, engine_path: str) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate input/output GPU buffers."""
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.bindings: list[int] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.prod(shape) * np.dtype(dtype).itemsize

            # Allocate device memory
            err, device_mem = cudart.cudaMalloc(size)
            binding = {"name": name, "dtype": dtype, "shape": shape, "device": device_mem}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            self.context.set_tensor_address(name, device_mem)

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on a single input."""
        # Copy input to device
        cudart.cudaMemcpy(
            self.inputs[0]["device"],
            input_array.ctypes.data,
            input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        # Execute
        self.context.execute_async_v3(0)

        # Copy output from device
        output = np.empty(self.outputs[0]["shape"], dtype=self.outputs[0]["dtype"])
        cudart.cudaMemcpy(
            output.ctypes.data,
            self.outputs[0]["device"],
            output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

        return output

    def __del__(self) -> None:
        """Free GPU memory."""
        for buf in self.inputs + self.outputs:
            cudart.cudaFree(buf["device"])
```

### Using ONNX Runtime with TensorRT Backend

If you want TensorRT performance without managing raw CUDA memory, use ONNX Runtime's TensorRT execution provider:

```python
import onnxruntime as ort


def create_tensorrt_session(
    onnx_path: str,
    fp16: bool = True,
    max_workspace_size: int = 4 * 1024 * 1024 * 1024,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with TensorRT backend.

    This is the simplest way to use TensorRT — ONNX Runtime handles
    engine building and caching automatically.
    """
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": max_workspace_size,
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache/",
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(onnx_path, providers=providers)
    return session
```

This approach is recommended for most projects — it gives TensorRT performance with the ONNX Runtime API you already know, and automatically falls back to CUDA or CPU if TensorRT cannot handle a layer.

## INT8 Calibration

INT8 provides the fastest inference but requires calibration data to preserve accuracy.

### Calibration with trtexec

```bash
# Generate calibration cache from a dataset
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_int8.engine \
    --int8 \
    --calib=calibration_cache.bin \
    --calibBatchSize=32
```

### Python Calibration

```python
import numpy as np
import tensorrt as trt


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration using representative images."""

    def __init__(
        self,
        calibration_images: list[np.ndarray],
        batch_size: int = 8,
        cache_file: str = "calibration.cache",
    ) -> None:
        super().__init__()
        self.images = calibration_images
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0

        # Allocate device memory for one batch
        self.batch_data = np.zeros(
            (batch_size, *calibration_images[0].shape), dtype=np.float32
        )
        _, self.device_input = cudart.cudaMalloc(self.batch_data.nbytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.current_index >= len(self.images):
            return None

        end = min(self.current_index + self.batch_size, len(self.images))
        batch = self.images[self.current_index : end]
        self.current_index = end

        # Pad if necessary
        self.batch_data[: len(batch)] = np.stack(batch)

        cudart.cudaMemcpy(
            self.device_input,
            self.batch_data.ctypes.data,
            self.batch_data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        from pathlib import Path

        cache = Path(self.cache_file)
        if cache.exists():
            return cache.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        from pathlib import Path

        Path(self.cache_file).write_bytes(cache)
```

## Pydantic Configuration

```python
from pydantic import BaseModel, Field


class TensorRTBuildConfig(BaseModel, frozen=True):
    """TensorRT engine build configuration."""

    onnx_path: str = Field(description="Path to slimmed ONNX model")
    engine_path: str = Field(description="Output engine path")
    fp16: bool = Field(default=True, description="Enable FP16 precision")
    int8: bool = Field(default=False, description="Enable INT8 precision")
    max_workspace_gb: int = Field(default=4, ge=1, le=32)
    max_batch_size: int = Field(default=1, ge=1)
    min_shape: tuple[int, ...] = Field(default=(1, 3, 640, 640))
    opt_shape: tuple[int, ...] = Field(default=(1, 3, 640, 640))
    max_shape: tuple[int, ...] = Field(default=(1, 3, 640, 640))
    calibration_cache: str | None = Field(default=None, description="INT8 calibration cache path")


class TensorRTInferenceConfig(BaseModel, frozen=True):
    """TensorRT inference configuration."""

    engine_path: str = Field(description="Path to compiled engine")
    device_id: int = Field(default=0, ge=0)
```

```yaml
# configs/tensorrt.yaml
tensorrt:
  onnx_path: models/model.onnx
  engine_path: models/model.engine
  fp16: true
  int8: false
  max_workspace_gb: 4
  max_batch_size: 8
  min_shape: [1, 3, 640, 640]
  opt_shape: [8, 3, 640, 640]
  max_shape: [32, 3, 640, 640]
```

## Integration with pixi

```toml
# pixi.toml — TensorRT tasks
[tasks]
# Full export pipeline: PyTorch → ONNX → OnnxSlim → TensorRT
export-onnx = "python -m my_project.export --format onnx"
export-trt = { cmd = "python -m my_project.export --format tensorrt", depends-on = ["export-onnx"] }

# Build engine with trtexec
trt-build = """trtexec \
    --onnx=models/model.onnx \
    --saveEngine=models/model.engine \
    --fp16"""

trt-build-dynamic = """trtexec \
    --onnx=models/model.onnx \
    --saveEngine=models/model.engine \
    --fp16 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x640x640"""

trt-benchmark = """trtexec \
    --loadEngine=models/model.engine \
    --iterations=1000 \
    --warmUp=500"""
```

## Benchmarking: ONNX Runtime vs TensorRT

```python
import time

import numpy as np
import onnxruntime as ort

from loguru import logger


def compare_backends(
    onnx_path: str,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    num_warmup: int = 50,
    num_runs: int = 500,
) -> None:
    """Compare ONNX Runtime CUDA vs TensorRT execution providers."""
    dummy = np.random.randn(*input_shape).astype(np.float32)

    results = {}
    for provider_name, providers in [
        ("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("TensorRT", [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]),
    ]:
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(num_warmup):
            session.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append((time.perf_counter() - start) * 1000)

        results[provider_name] = {
            "mean_ms": np.mean(times),
            "p99_ms": np.percentile(times, 99),
            "fps": 1000.0 / np.mean(times),
        }

        logger.info(
            "{}: {:.2f} ms mean, {:.2f} ms p99, {:.0f} FPS",
            provider_name,
            results[provider_name]["mean_ms"],
            results[provider_name]["p99_ms"],
            results[provider_name]["fps"],
        )

    speedup = results["CUDA"]["mean_ms"] / results["TensorRT"]["mean_ms"]
    logger.info("TensorRT speedup over CUDA: {:.2f}x", speedup)
```

## Dockerfile for TensorRT

```dockerfile
# Use NVIDIA's TensorRT container as base
FROM nvcr.io/nvidia/tensorrt:24.08-py3

WORKDIR /app

# Install project dependencies
COPY pixi.toml pixi.lock ./
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy model and application
COPY models/ models/
COPY src/ src/

# Build TensorRT engine at container startup (GPU-specific)
# Or copy a pre-built engine for the target GPU
ENTRYPOINT ["python", "-m", "my_project.serve"]
```

Note: TensorRT engines are GPU-architecture-specific. An engine built on A100 will not work on T4. Either build engines at container startup or build separate images per GPU target.

## Best Practices

1. **Always export to ONNX first** — use the ONNX skill pipeline (export → slim → validate) before TensorRT conversion.
2. **Use FP16 as the default precision** — it halves memory usage with negligible accuracy loss on most models.
3. **Use ONNX Runtime TensorRT EP for simplicity** — it manages engine building and caching automatically; use raw TensorRT API only when you need maximum control.
4. **Cache built engines** — engine builds are slow (minutes); cache them by GPU architecture and rebuild only when the model changes.
5. **Set dynamic shapes with realistic profiles** — `opt_shape` should match your most common batch size and resolution.
6. **Validate accuracy after conversion** — compare TensorRT outputs against ONNX/PyTorch to catch precision issues.
7. **Profile with `trtexec`** — use `--dumpProfile` to identify slow layers.
8. **Pin TensorRT and CUDA versions** — document the exact versions in your Dockerfile and README.
9. **Use INT8 only with calibration data** — uncalibrated INT8 can cause significant accuracy loss.
10. **Build per-GPU-architecture** — engines are not portable across GPU architectures (e.g., A100 vs T4).

## Anti-Patterns to Avoid

- ❌ Exporting PyTorch directly to TensorRT without going through ONNX — use the ONNX skill pipeline first.
- ❌ Deploying raw ONNX models to TensorRT without running `onnxslim.slim()` — slimming removes redundant ops that confuse the TensorRT optimizer.
- ❌ Assuming TensorRT engines are portable — they are tied to the GPU architecture and TensorRT version they were built on.
- ❌ Using INT8 without calibration data — this causes significant accuracy degradation.
- ❌ Setting `max_workspace_size` too low — TensorRT needs workspace memory for kernel auto-tuning; start with 4 GB.
- ❌ Skipping warmup in benchmarks — the first few inferences include JIT compilation and memory allocation overhead.
- ❌ Using TensorRT on CPUs or non-NVIDIA GPUs — it only runs on NVIDIA hardware; use ONNX Runtime for cross-platform deployment.
