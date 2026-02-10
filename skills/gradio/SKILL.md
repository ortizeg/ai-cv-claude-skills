---
name: gradio
description: >
  Gradio patterns for building interactive ML demos and model serving UIs.
  Covers gr.Interface, gr.Blocks layouts, image/video/text inputs and outputs,
  model serving with gr.load, custom components, flagging and feedback collection,
  deployment to Hugging Face Spaces, and integration with PyTorch/ONNX models.
---

# Gradio Skill

You are building Gradio applications for interactive ML demos and model serving UIs. Follow these patterns exactly.

## Core Philosophy

Gradio provides the fastest path from a trained model to an interactive web demo. Every demo in this framework uses Gradio 4.x with typed interfaces, Pydantic-validated configurations, and Loguru-based logging. Use `gr.Blocks` for complex layouts and `gr.Interface` for simple single-function demos. Never expose raw model internals to the UI layer.

## Interface Creation

### Simple Interface with gr.Interface

Use `gr.Interface` when wrapping a single prediction function with clearly defined inputs and outputs. This is the fastest way to prototype a demo.

```python
"""Simple image classification demo with gr.Interface."""

from __future__ import annotations

import gradio as gr
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field


class ClassificationConfig(BaseModel, frozen=True):
    """Configuration for classification demo."""

    model_path: str = "models/classifier.onnx"
    labels_path: str = "models/labels.txt"
    top_k: int = Field(default=5, ge=1, le=20)
    image_size: tuple[int, int] = (224, 224)


def classify_image(
    image: Image.Image,
    config: ClassificationConfig = ClassificationConfig(),
) -> dict[str, float]:
    """Run classification on a single image and return top-k predictions."""
    logger.info("Received image of size {}", image.size)
    preprocessed = preprocess(image, config.image_size)
    predictions = run_model(preprocessed, config.model_path)
    labels = load_labels(config.labels_path)

    top_indices = np.argsort(predictions)[-config.top_k :][::-1]
    results = {labels[i]: float(predictions[i]) for i in top_indices}
    logger.info("Top prediction: {} ({:.3f})", list(results.keys())[0], list(results.values())[0])
    return results


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Image Classifier",
    description="Upload an image to classify it using the trained model.",
    examples=[["examples/cat.jpg"], ["examples/dog.jpg"]],
    cache_examples=True,
)
```

### Complex Layouts with gr.Blocks

Use `gr.Blocks` for multi-step workflows, side-by-side comparisons, tabs, and conditional visibility. This is the standard for production demos.

```python
"""Object detection demo with gr.Blocks layout."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field


class DetectionConfig(BaseModel, frozen=True):
    """Detection demo configuration."""

    model_path: str = "models/detector.onnx"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=500)
    device: str = "cpu"


def build_detection_demo(config: DetectionConfig | None = None) -> gr.Blocks:
    """Build the detection demo application."""
    config = config or DetectionConfig()
    logger.info("Building detection demo with model: {}", config.model_path)

    with gr.Blocks(title="Object Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Object Detection Demo")
        gr.Markdown("Upload an image or use the webcam to detect objects in real time.")

        with gr.Tab("Image Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Input Image")
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.confidence_threshold,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    detect_btn = gr.Button("Detect Objects", variant="primary")

                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Detections")
                    output_json = gr.JSON(label="Detection Results")

            detect_btn.click(
                fn=run_detection,
                inputs=[input_image, confidence_slider],
                outputs=[output_image, output_json],
            )

        with gr.Tab("Webcam"):
            webcam_input = gr.Image(sources=["webcam"], type="pil", label="Webcam Feed")
            webcam_output = gr.Image(type="pil", label="Live Detections")
            webcam_input.stream(
                fn=run_detection_realtime,
                inputs=[webcam_input],
                outputs=[webcam_output],
            )

        with gr.Tab("Batch Processing"):
            batch_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload Multiple Images",
            )
            batch_output = gr.Gallery(label="Results", columns=3)
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_btn.click(
                fn=run_batch_detection,
                inputs=[batch_input, confidence_slider],
                outputs=[batch_output],
            )

    return demo
```

## Image, Video, and Text Inputs and Outputs

### Multi-Modal Input/Output Patterns

```python
"""Multi-modal Gradio interface patterns."""

from __future__ import annotations

import gradio as gr
from loguru import logger


def build_multimodal_demo() -> gr.Blocks:
    """Build a demo with image, video, and text I/O."""
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Modal Analysis")

        with gr.Tab("Image Captioning"):
            img_input = gr.Image(type="pil", label="Upload Image")
            caption_output = gr.Textbox(label="Generated Caption", lines=3)
            img_input.change(fn=generate_caption, inputs=[img_input], outputs=[caption_output])

        with gr.Tab("Video Analysis"):
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Video(label="Annotated Video")
            frame_gallery = gr.Gallery(label="Key Frames", columns=4)
            analysis_json = gr.JSON(label="Analysis Results")

            analyze_btn = gr.Button("Analyze Video", variant="primary")
            analyze_btn.click(
                fn=analyze_video,
                inputs=[video_input],
                outputs=[video_output, frame_gallery, analysis_json],
            )

        with gr.Tab("Visual Question Answering"):
            with gr.Row():
                vqa_image = gr.Image(type="pil", label="Image")
                with gr.Column():
                    vqa_question = gr.Textbox(label="Question", placeholder="What is in this image?")
                    vqa_answer = gr.Textbox(label="Answer", interactive=False)
                    vqa_btn = gr.Button("Ask", variant="primary")

            vqa_btn.click(
                fn=answer_question,
                inputs=[vqa_image, vqa_question],
                outputs=[vqa_answer],
            )

    return demo
```

## Model Serving with gr.load

### Loading Models from Hugging Face Hub

Use `gr.load` to quickly create a demo from a Hugging Face model without writing inference code.

```python
"""Serve Hugging Face models directly with gr.load."""

from __future__ import annotations

import gradio as gr
from loguru import logger

# Load a model directly from the Hugging Face Hub
logger.info("Loading model from Hugging Face Hub")
demo = gr.load(
    name="facebook/detr-resnet-50",
    src="models",
    title="DETR Object Detection",
    description="Detect objects using DETR (DEtection TRansformer).",
)

# Load a Hugging Face Space as a component
image_gen = gr.load(
    name="stabilityai/stable-diffusion-2",
    src="models",
)
```

### Serving Custom PyTorch and ONNX Models

```python
"""Serve a custom ONNX model through Gradio."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field


class ONNXModelConfig(BaseModel, frozen=True):
    """ONNX model serving configuration."""

    model_path: Path = Path("models/model.onnx")
    input_size: tuple[int, int] = (640, 640)
    providers: list[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


class ModelServer:
    """Wraps an ONNX model for Gradio serving."""

    def __init__(self, config: ONNXModelConfig) -> None:
        self.config = config
        logger.info("Loading ONNX model from {}", config.model_path)
        self.session = ort.InferenceSession(
            str(config.model_path),
            providers=config.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Model loaded successfully with providers: {}", config.providers)

    def predict(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """Run inference and return annotated image and results dict."""
        logger.debug("Preprocessing image of size {}", image.size)
        input_array = self._preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_array})
        detections = self._postprocess(outputs, image.size)

        annotated = self._draw_detections(image, detections)
        results = {"num_detections": len(detections), "detections": detections}
        logger.info("Found {} detections", len(detections))
        return annotated, results

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize input image."""
        resized = image.resize(self.config.input_size)
        array = np.array(resized, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))[np.newaxis, ...]

    def _postprocess(self, outputs: list, original_size: tuple[int, int]) -> list[dict]:
        """Convert raw model outputs to detection dicts."""
        # Implementation depends on model architecture
        ...

    def _draw_detections(self, image: Image.Image, detections: list[dict]) -> Image.Image:
        """Draw bounding boxes on the image."""
        # Implementation uses PIL.ImageDraw
        ...


def create_onnx_demo(config: ONNXModelConfig | None = None) -> gr.Blocks:
    """Create a Gradio demo serving an ONNX model."""
    config = config or ONNXModelConfig()
    server = ModelServer(config)

    with gr.Blocks(title="ONNX Model Demo") as demo:
        gr.Markdown("# Custom ONNX Model Demo")
        with gr.Row():
            input_img = gr.Image(type="pil", label="Input")
            output_img = gr.Image(type="pil", label="Output")
        results_json = gr.JSON(label="Results")
        run_btn = gr.Button("Run Inference", variant="primary")

        run_btn.click(
            fn=server.predict,
            inputs=[input_img],
            outputs=[output_img, results_json],
        )

    return demo
```

## Custom Components and Layouts

### Reusable Component Patterns

```python
"""Reusable Gradio component patterns."""

from __future__ import annotations

import gradio as gr
from loguru import logger


def create_model_selector(
    models: dict[str, str],
    default: str | None = None,
) -> gr.Dropdown:
    """Create a model selection dropdown with display names."""
    choices = list(models.keys())
    return gr.Dropdown(
        choices=choices,
        value=default or choices[0],
        label="Select Model",
        info="Choose a model for inference",
    )


def create_preprocessing_controls() -> tuple[gr.Slider, gr.Slider, gr.Checkbox]:
    """Create standard image preprocessing controls."""
    brightness = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Brightness")
    contrast = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Contrast")
    grayscale = gr.Checkbox(value=False, label="Convert to Grayscale")
    return brightness, contrast, grayscale


def build_comparison_layout() -> gr.Blocks:
    """Build a side-by-side model comparison layout."""
    with gr.Blocks() as demo:
        gr.Markdown("# Model Comparison")
        input_image = gr.Image(type="pil", label="Input Image")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model A")
                model_a_selector = create_model_selector(
                    {"YOLOv8-n": "yolov8n", "YOLOv8-s": "yolov8s"},
                    default="YOLOv8-n",
                )
                output_a = gr.Image(type="pil", label="Model A Output")
                metrics_a = gr.JSON(label="Model A Metrics")

            with gr.Column():
                gr.Markdown("### Model B")
                model_b_selector = create_model_selector(
                    {"YOLOv8-m": "yolov8m", "YOLOv8-l": "yolov8l"},
                    default="YOLOv8-m",
                )
                output_b = gr.Image(type="pil", label="Model B Output")
                metrics_b = gr.JSON(label="Model B Metrics")

        compare_btn = gr.Button("Compare Models", variant="primary")
        compare_btn.click(
            fn=compare_models,
            inputs=[input_image, model_a_selector, model_b_selector],
            outputs=[output_a, metrics_a, output_b, metrics_b],
        )

    return demo
```

## Flagging and Feedback Collection

### Custom Flagging Callback

Use flagging to collect user feedback on model predictions for dataset improvement and active learning.

```python
"""Custom flagging callback for ML feedback collection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from loguru import logger
from pydantic import BaseModel, Field


class FlaggedSample(BaseModel, frozen=True):
    """A flagged sample with metadata."""

    timestamp: str
    flag_reason: str
    input_hash: str
    prediction: str
    user_correction: str | None = None


class MLFlaggingCallback(gr.FlaggingCallback):
    """Custom flagging callback that saves structured feedback."""

    def __init__(self, output_dir: str = "flagged_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Flagging callback initialized, saving to {}", self.output_dir)

    def setup(self, components: list, flagging_dir: str | Path) -> None:
        """Set up the flagging directory."""
        self.flagging_dir = Path(flagging_dir)
        self.flagging_dir.mkdir(parents=True, exist_ok=True)

    def flag(
        self,
        flag_data: list,
        flag_option: str = "incorrect",
        username: str | None = None,
    ) -> int:
        """Save a flagged sample with structured metadata."""
        sample = FlaggedSample(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            flag_reason=flag_option,
            input_hash=str(hash(str(flag_data[0]))),
            prediction=str(flag_data[1]) if len(flag_data) > 1 else "",
            user_correction=str(flag_data[2]) if len(flag_data) > 2 else None,
        )
        output_path = self.output_dir / f"flag_{sample.timestamp}.json"
        output_path.write_text(sample.model_dump_json(indent=2))
        logger.info("Flagged sample saved: {} (reason: {})", output_path, flag_option)
        return 1


# Usage in a demo
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    flagging_callback=MLFlaggingCallback(output_dir="flagged_data"),
    flagging_options=["incorrect", "low_confidence", "interesting"],
)
```

## Deployment

### Sharing and Hugging Face Spaces

```python
"""Deployment patterns for Gradio demos."""

from __future__ import annotations

import gradio as gr
from loguru import logger


def launch_demo(demo: gr.Blocks, share: bool = False) -> None:
    """Launch the Gradio demo with production settings."""
    logger.info("Launching Gradio demo (share={})", share)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True,
        max_threads=10,
        auth=None,  # Set ("user", "pass") for basic auth
    )


def launch_with_auth(demo: gr.Blocks) -> None:
    """Launch with authentication for internal deployments."""
    logger.info("Launching Gradio demo with authentication")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=[("admin", "secure_password")],
        auth_message="Enter credentials to access the demo.",
    )


def mount_on_fastapi(demo: gr.Blocks) -> None:
    """Mount Gradio inside a FastAPI application."""
    from fastapi import FastAPI

    app = FastAPI(title="ML Demo API")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    app = gr.mount_gradio_app(app, demo, path="/demo")
    logger.info("Gradio demo mounted at /demo")
```

### Hugging Face Spaces Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 7860

CMD ["uv", "run", "python", "-m", "src.app", "--server-name", "0.0.0.0", "--server-port", "7860"]
```

### Hugging Face Spaces app.py entry point

```python
"""Hugging Face Spaces entry point."""

from __future__ import annotations

import gradio as gr
from loguru import logger

from src.demo import build_detection_demo
from src.config import DetectionConfig


logger.info("Starting Hugging Face Spaces demo")
config = DetectionConfig()
demo = build_detection_demo(config)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Integration with PyTorch Models

### PyTorch Lightning Model in Gradio

```python
"""Serve a PyTorch Lightning model through Gradio."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms


class TorchModelConfig(BaseModel, frozen=True):
    """PyTorch model serving configuration."""

    checkpoint_path: Path = Path("checkpoints/best.ckpt")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: tuple[int, int] = (224, 224)
    class_names: list[str] = Field(default_factory=list)


def load_pytorch_model(config: TorchModelConfig) -> torch.nn.Module:
    """Load a PyTorch model from checkpoint."""
    logger.info("Loading PyTorch model from {}", config.checkpoint_path)
    model = torch.load(config.checkpoint_path, map_location=config.device)
    model.eval()
    logger.info("Model loaded on device: {}", config.device)
    return model


def create_pytorch_demo(config: TorchModelConfig | None = None) -> gr.Blocks:
    """Create a Gradio demo for a PyTorch model."""
    config = config or TorchModelConfig()
    model = load_pytorch_model(config)

    transform = transforms.Compose([
        transforms.Resize(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.inference_mode()
    def predict(image: Image.Image) -> dict[str, float]:
        """Run inference on a single image."""
        tensor = transform(image).unsqueeze(0).to(config.device)
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_idx = torch.topk(probabilities, 5)
        results = {}
        for prob, idx in zip(top5_prob, top5_idx):
            label = config.class_names[idx] if config.class_names else f"class_{idx}"
            results[label] = float(prob)

        logger.info("Top prediction: {} ({:.3f})", list(results.keys())[0], list(results.values())[0])
        return results

    with gr.Blocks(title="PyTorch Model Demo") as demo:
        gr.Markdown("# PyTorch Model Demo")
        with gr.Row():
            input_img = gr.Image(type="pil", label="Upload Image")
            output_label = gr.Label(num_top_classes=5, label="Predictions")
        predict_btn = gr.Button("Classify", variant="primary")
        predict_btn.click(fn=predict, inputs=[input_img], outputs=[output_label])

    return demo
```

## Anti-Patterns

- **Never load models inside Gradio callback functions** -- load models once at startup and reference them via closure or a server class. Reloading on every request causes massive latency.
- **Never use `gr.Interface` for complex multi-step workflows** -- use `gr.Blocks` with explicit layout control for anything beyond a single input-output function.
- **Never expose raw tensors or numpy arrays to Gradio outputs** -- always convert to PIL Images, JSON-serializable dicts, or strings before returning.
- **Never hardcode file paths in Gradio demos** -- use Pydantic config models to make paths configurable for different environments.
- **Never skip input validation** -- validate image sizes, file types, and parameter ranges before feeding data to the model.
- **Never use `share=True` in production deployments** -- use proper hosting (Hugging Face Spaces, Docker, or a reverse proxy) instead of Gradio's temporary share links.
- **Never block the main thread with long-running inference** -- use `gr.Progress` for progress tracking and consider async patterns for heavy workloads.
- **Never ignore logging** -- use Loguru for structured logging in all prediction functions, model loading, and error handling.

## Integration with Other Skills

- **Pydantic Strict** -- All configuration models follow frozen BaseModel patterns with validators.
- **Loguru** -- Structured logging in model loading, prediction functions, and flagging callbacks.
- **Hugging Face** -- Load pretrained models from the Hub and deploy demos to Hugging Face Spaces.
- **FastAPI** -- Mount Gradio demos inside FastAPI applications for combined API + demo serving.
- **ONNX** -- Serve optimized ONNX models through Gradio for low-latency inference.
- **PyTorch Lightning** -- Load Lightning checkpoints and serve trained models through Gradio.
- **Testing** -- Test Gradio components with the Gradio test client and pytest.
