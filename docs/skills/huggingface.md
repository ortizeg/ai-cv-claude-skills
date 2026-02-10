# Hugging Face

The Hugging Face skill provides expert patterns for the Hugging Face ecosystem — Transformers, Datasets, PEFT, and the Model Hub — covering pretrained model loading, fine-tuning, and inference.

**Skill directory:** `skills/huggingface/`

## Purpose

The Hugging Face ecosystem is the standard toolkit for working with pretrained models. This skill encodes best practices for loading models with AutoModel classes and version pinning, preprocessing datasets with batched mapping, fine-tuning with the Trainer API, parameter-efficient fine-tuning with LoRA, and publishing models with model cards.

## When to Use

Use this skill whenever you need to:

- Load pretrained vision models (ResNet, ViT, DETR, CLIP) or text models (BERT, RoBERTa)
- Fine-tune models on custom datasets with the Trainer API
- Use PEFT/LoRA for parameter-efficient adaptation with limited data
- Build quick inference pipelines for classification, detection, or zero-shot tasks
- Publish models and datasets to the Hugging Face Hub

## Key Patterns

### AutoModel Loading

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(
    "microsoft/resnet-50",
    revision="main",
)

model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    revision="main",
    torch_dtype=torch.float32,
)
```

### Fine-Tuning with Trainer

```python
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="outputs/finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    fp16=True,
    eval_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
```

### PEFT / LoRA

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)

model = get_peft_model(base_model, lora_config)
# Trains <1% of parameters
```

## Anti-Patterns to Avoid

- Do not hardcode model class names -- use `AutoModel`, `AutoTokenizer` for flexibility
- Do not download models inside training loops -- load once, cache with `cache_dir`
- Do not skip `revision` parameter -- pin model versions for reproducibility
- Do not fine-tune all parameters with limited data -- use PEFT/LoRA

## Combines Well With

- **PyTorch Lightning** -- Custom training loops wrapping HF models
- **ONNX** -- Export HF models for optimized inference
- **W&B** -- Experiment tracking via `report_to=["wandb"]`
- **AWS SageMaker** -- Deploy HF models with SageMaker HF DLC
- **FastAPI** -- Serve HF pipelines behind async endpoints

## Full Reference

See [`skills/huggingface/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/huggingface/SKILL.md) for complete patterns including datasets, pipelines, model hub publishing, and quantization.
