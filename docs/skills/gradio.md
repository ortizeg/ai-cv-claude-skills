# Gradio

The Gradio skill provides expert patterns for building ML model demos and interactive interfaces with Gradio, covering Interface/Blocks layouts, input/output components, and deployment to Hugging Face Spaces.

**Skill directory:** `skills/gradio/`

## Purpose

Gradio is the standard tool for quickly building interactive demos for ML models. This skill encodes best practices for structuring Gradio applications: choosing between `gr.Interface` and `gr.Blocks`, handling image/video/text inputs, integrating with PyTorch and ONNX models, and deploying to Hugging Face Spaces.

## When to Use

Use this skill whenever you need to:

- Build interactive demos for trained models
- Create quick prototypes for stakeholder review
- Deploy model demos to Hugging Face Spaces
- Build custom UIs with multi-step workflows using Blocks

## Key Patterns

### Quick Interface

```python
import gradio as gr

def classify(image):
    predictions = model(preprocess(image))
    return {labels[i]: float(p) for i, p in enumerate(predictions)}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classifier",
)
demo.launch()
```

### Blocks Layout

```python
with gr.Blocks() as demo:
    gr.Markdown("# Object Detection")
    with gr.Row():
        input_image = gr.Image(type="pil")
        output_image = gr.Image()
    confidence = gr.Slider(0, 1, value=0.5, label="Confidence")
    btn = gr.Button("Detect")
    btn.click(detect, [input_image, confidence], output_image)
```

## Anti-Patterns to Avoid

- Do not load models inside prediction functions -- load once at module level
- Do not use `share=True` for production deployments -- use proper hosting
- Do not skip input validation -- validate image formats and sizes

## Combines Well With

- **Hugging Face** -- Load models from Hub, deploy to Spaces
- **FastAPI** -- Embed Gradio as a sub-app in larger services
- **ONNX / TensorRT** -- Serve optimized models through Gradio interfaces

## Full Reference

See [`skills/gradio/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/gradio/SKILL.md) for complete patterns.
