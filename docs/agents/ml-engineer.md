# ML Engineer Agent

Advisory agent specialized in model architecture, training pipelines, experiment management, and CV-specific optimization.

## Purpose

The ML Engineer reviews and suggests improvements for:

- **Model Architecture** -- layer choices, regularization, transfer learning, bottleneck identification
- **Training Pipelines** -- loss functions, optimizers, LR schedules, gradient management
- **Experiment Management** -- hyperparameter selection, metric tracking, reproducibility
- **CV-Specific Knowledge** -- augmentation strategies, multi-scale training, detection/segmentation patterns

## Strictness Level

**Advisory** -- suggests but does not block.

## Common Pitfalls It Catches

### Missing Gradient Clipping

```python
# ❌ Risk of gradient explosion
optimizer.step()

# ✅ Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Wrong Normalization

```python
# ❌ Generic normalization
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# ✅ ImageNet normalization for pretrained models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Batch Norm After Dropout

```python
# ❌ Wrong order -- BN after dropout is unstable
nn.Sequential(nn.Conv2d(64, 128, 3), nn.Dropout(0.5), nn.BatchNorm2d(128))

# ✅ Correct order -- BN before dropout
nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5))
```

## Recommendations by Task

| Task | Architecture | Loss | Augmentation |
|------|-------------|------|-------------|
| Object Detection | FCOS, DETR + FPN | Focal Loss | Mosaic, MixUp |
| Segmentation | U-Net, DeepLab | Dice + CE | Elastic, ColorJitter |
| Classification | ResNet, EfficientNet | Label Smoothing CE | MixUp, CutMix |

## When to Use

- Designing new model architectures
- Debugging training instability (loss spikes, NaN gradients)
- Optimizing training performance (speed, memory)
- Reviewing experiment setup for reproducibility
