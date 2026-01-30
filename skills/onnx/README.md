# ONNX Model Export and Inference Skill

## Purpose

This skill provides guidance on exporting PyTorch models to ONNX format and running optimized inference with ONNX Runtime. ONNX enables production deployment with significant speed improvements over native PyTorch inference, hardware-agnostic execution, and standardized model packaging.

## Usage

Reference this skill when:

- Exporting a trained PyTorch model for production inference.
- Configuring dynamic axes for variable input sizes.
- Setting up ONNX Runtime inference sessions with GPU support.
- Optimizing models with graph optimization or quantization.
- Validating that ONNX outputs match PyTorch outputs.
- Building inference APIs with FastAPI and ONNX Runtime.
- Benchmarking inference performance (PyTorch vs ONNX).
- Debugging export failures or runtime errors.

## Setup

```bash
pip install onnx onnxruntime-gpu onnxslim  # or onnxruntime for CPU-only
```

## What This Skill Covers

- Basic and advanced model export with `torch.onnx.export`.
- **OnnxSlim** as a required post-export optimization step.
- Pydantic-validated export configuration.
- Dynamic axes for variable batch size and image dimensions.
- Input/output inspection and multiple input/output models.
- ONNX Runtime inference with execution providers (CPU, CUDA, TensorRT).
- Graph optimization and INT8/FP16 quantization.
- Output validation comparing PyTorch and ONNX.
- FastAPI integration for serving models.
- Performance benchmarking utilities.
- Common pitfalls and their solutions.

## Benefits

- 2-10x faster inference compared to native PyTorch.
- Hardware-agnostic deployment (CPU, GPU, mobile, edge).
- No PyTorch dependency required at inference time.
- Built-in quantization for further speedup on CPU.
- Standardized format supported by major cloud providers.
- Graph optimization automatically fuses operations.

See `SKILL.md` for complete documentation and code examples.
