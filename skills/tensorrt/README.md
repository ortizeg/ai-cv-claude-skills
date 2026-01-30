# TensorRT Skill

The TensorRT Skill covers NVIDIA TensorRT engine building and deployment for maximum inference performance on NVIDIA GPUs. TensorRT applies aggressive graph optimizations — layer fusion, kernel auto-tuning, precision calibration — that go beyond what ONNX Runtime's CUDA provider offers, delivering 2-6x additional speedup on the same hardware.

This skill depends on the ONNX skill. The required pipeline is: PyTorch → ONNX export → OnnxSlim → TensorRT engine build. TensorRT does not consume PyTorch models directly, and slimming the ONNX model before conversion produces better engines.

TensorRT is not required for every project. Use it when deploying to known NVIDIA hardware where latency is critical. For cross-platform or CPU deployment, the ONNX skill covers everything you need.

## When to Use

- When you need the lowest possible inference latency on NVIDIA GPUs.
- When deploying to a known GPU architecture (A100, T4, L4, etc.) where you can build GPU-specific engines.
- When FP16 or INT8 precision is acceptable and you want automatic precision calibration.
- When throughput (inferences/second) is a primary deployment metric.
- When deploying in NVIDIA containers (NGC TensorRT images, Triton Inference Server).

## Key Features

- **trtexec CLI** — convert ONNX models to TensorRT engines with a single command, including FP16/INT8 and dynamic shapes.
- **Python Builder API** — programmatic engine building with full control over optimization profiles, precision, and workspace.
- **ONNX Runtime TensorRT EP** — use TensorRT performance with the ONNX Runtime API for automatic engine caching and CUDA fallback.
- **Dynamic shapes** — min/opt/max shape profiles for variable batch sizes and image dimensions.
- **INT8 calibration** — entropy-based calibration with representative data for INT8 inference.
- **Pydantic configuration** — typed config models for build and inference settings.
- **Benchmarking utilities** — compare ONNX Runtime CUDA vs TensorRT side-by-side.

## Related Skills

- **[ONNX](../onnx/)** — prerequisite; provides the export and slimming pipeline that produces the ONNX model TensorRT consumes.
- **[Docker CV](../docker-cv/)** — TensorRT inference containers use NVIDIA NGC base images with TensorRT pre-installed.
- **[GCP](../gcp/)** — Vertex AI supports TensorRT-optimized containers for inference endpoints.
- **[PyTorch Lightning](../pytorch-lightning/)** — training produces checkpoints that are exported through the ONNX → TensorRT pipeline.
