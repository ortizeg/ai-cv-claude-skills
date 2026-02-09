# Hugging Face Skill

The Hugging Face Skill provides expert patterns for working with the Hugging Face ecosystem — Transformers, Datasets, Tokenizers, PEFT, and the Model Hub. It covers pretrained model loading with AutoModel classes, dataset preprocessing, fine-tuning with the Trainer API, parameter-efficient fine-tuning with LoRA, pipeline inference, model publishing, and quantization for optimized deployment.

## Purpose

When you need to use pretrained models for vision or NLP tasks — whether for classification, detection, zero-shot learning, or fine-tuning on custom data — the Hugging Face ecosystem is the standard toolkit. This skill encodes best practices for loading models safely with version pinning, preprocessing datasets efficiently with batched mapping, fine-tuning with proper evaluation metrics and early stopping, and publishing models with complete model cards.

## When to Use

- When loading pretrained vision models (ResNet, ViT, DETR, CLIP) or text models (BERT, RoBERTa).
- When fine-tuning models on custom datasets with the Trainer API.
- When using PEFT/LoRA for parameter-efficient adaptation with limited data.
- When building quick inference pipelines for classification, detection, or zero-shot tasks.
- When publishing models and datasets to the Hugging Face Hub.
- When quantizing models to 4-bit or 8-bit for faster inference.

## Key Features

- **AutoModel loading** — type-safe model loading with revision pinning and device mapping.
- **Datasets library** — efficient loading, preprocessing, and batched tokenization.
- **Trainer API** — standardized fine-tuning with evaluation, early stopping, and checkpoint management.
- **PEFT / LoRA** — parameter-efficient fine-tuning that trains <1% of parameters.
- **Pipeline API** — quick inference for image classification, object detection, and zero-shot tasks.
- **Hub publishing** — model upload with model cards, metrics, and licensing.
- **Quantization** — 4-bit and 8-bit model loading for memory-efficient inference.

## Related Skills

- **[PyTorch Lightning](../pytorch-lightning/)** — custom training loops wrapping HF models in LightningModule.
- **[ONNX](../onnx/)** — export HF models for optimized inference with ONNX Runtime.
- **[W&B](../wandb/)** — experiment tracking integrated with HF Trainer via `report_to`.
- **[AWS SageMaker](../aws-sagemaker/)** — deploy HF models on SageMaker with HF Deep Learning Containers.
- **[FastAPI](../fastapi/)** — serve HF pipelines behind async REST endpoints.
