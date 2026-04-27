# Condensed: default used by AutoMM

Summary: This tutorial is a comprehensive hyperparameter reference for AutoGluon's AutoMM (MultiModalPredictor), covering all configurable parameters passed via `predictor.fit(hyperparameters={...})`. It details optimization settings (learning rate, optimizer types, layer-wise LR decay, LR schedules, warmup, early stopping, gradient clipping, PEFT methods including LoRA/IA3/BitFit, and model averaging via greedy/uniform soup), environment configuration (GPU setup, batch sizes with gradient accumulation, mixed precision, distributed strategies, torch.compile), model selection and configuration for text (HuggingFace), image (TIMM), CLIP, MMDetection, SAM, and tabular (FT-Transformer/MLP) backbones, data preprocessing (missing values, text normalization, numerical scaling, Mixup/Cutmix augmentation), and knowledge distillation parameters.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Optimization Hyperparameters

## Learning Rate & Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optim.lr` | `1.0e-4` | Learning rate |
| `optim.optim_type` | `"adamw"` | Optimizer: `"sgd"`, `"adam"`, `"adamw"` |
| `optim.weight_decay` | `1.0e-3` | Weight decay |

## Layer-wise Learning Rate Strategies

**`optim.lr_choice`** (default: `"layerwise_decay"`): Strategy for per-layer LR — `"layerwise_decay"`, `"two_stages"`, or `""` (uniform).

- **`optim.lr_decay`** (default: `0.9`): For layerwise decay — layer `i` gets LR `optim.lr * optim.lr_decay^(n-i)`. Set to `1` for uniform LR.
- **`optim.lr_mult`** (default: `1`): For two-stage — head layer gets `optim.lr * optim.lr_mult`, others get `optim.lr`.

## LR Schedule & Training Duration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optim.lr_schedule` | `"cosine_decay"` | Options: `"cosine_decay"`, `"polynomial_decay"`, `"linear_decay"` |
| `optim.max_epochs` | `10` | Max training epochs |
| `optim.max_steps` | `-1` (disabled) | Max steps; training stops at whichever limit is reached first |
| `optim.warmup_steps` | `0.1` | Fraction of steps for LR warmup (0 → `optim.lr`) |

## Early Stopping & Validation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optim.patience` | `10` | Checks with no improvement before stopping |
| `optim.val_check_interval` | `0.5` | Float (fraction of epoch) or int (every N batches) |

## Gradient Clipping & Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optim.gradient_clip_algorithm` | `"norm"` | `"norm"` or `"value"` |
| `optim.gradient_clip_val` | `1` | Clipping threshold |
| `optim.track_grad_norm` | `-1` (off) | Track p-norm of gradients; supports `'inf'`. AMP gradients are unscaled before logging |
| `optim.log_every_n_steps` | `10` | Logging frequency |

## Model Averaging

**`optim.top_k`** (default: `3`): Number of best checkpoints for averaging.

**`optim.top_k_average_method`** (default: `"greedy_soup"`):
- `"greedy_soup"`: Incrementally adds checkpoints best-to-worst, stops if performance drops
- `"uniform_soup"`: Averages all top-k checkpoints
- `"best"`: Uses single best checkpoint

## Parameter-Efficient Finetuning (PEFT)

**`optim.peft`** (default: `None` — full finetune):

| Option | Description |
|--------|-------------|
| `"bit_fit"` | Bias parameters only |
| `"norm_fit"` | Normalization + bias parameters |
| `"lora"` / `"lora_bias"` / `"lora_norm"` | LoRA adaptors (optionally + bias/norm) |
| `"ia3"` / `"ia3_bias"` / `"ia3_norm"` | IA3 algorithm (optionally + bias/norm) |

```python
# Full finetune (default)
predictor.fit(hyperparameters={"optim.peft": None})

# LoRA + bias
predictor.fit(hyperparameters={"optim.peft": "lora_bias"})
```

# AutoMM Environment & Model Configuration

## Environment

### GPU & Batch Size

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.num_gpus` | `-1` (all available) | Number of GPUs to use |
| `env.per_gpu_batch_size` | `8` | Batch size per GPU |
| `env.batch_size` | `128` | Effective batch size; gradient accumulation = `batch_size // (per_gpu_batch_size * num_gpus)` |
| `env.inference_batch_size_ratio` | `4` | Inference batch = `per_gpu_batch_size * ratio` |

### Precision & Performance

**`env.precision`** (default: `"16-mixed"`): Options — `64`, `32`, `"bf16-mixed"`, `"bf16-true"`, `"16-mixed"`, `"16-true"`. Mixed precision (`"16-mixed"`) can achieve +3x speedups on modern GPUs.

### Workers & Distribution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.num_workers` | `2` | Dataloader workers for training. More workers don't always help with `"ddp_spawn"` |
| `env.num_workers_inference` | `2` | Dataloader workers for prediction/evaluation |
| `env.strategy` | `"ddp_spawn"` | `"dp"`, `"ddp"` (script-based), `"ddp_spawn"` (spawn-based) |
| `env.accelerator` | `"auto"` | `"cpu"`, `"gpu"`, or `"auto"` (GPU prioritized) |

### torch.compile

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.compile.turn_on` | `False` | Enable `torch.compile`. **Recommended for large models/long training** |
| `env.compile.mode` | `"default"` | `"default"`, `"reduce-overhead"`, `"max-autotune"`, `"max-autotune-no-cudagraphs"` |
| `env.compile.dynamic` | `True` | Dynamic shape tracing; `False` assumes static input shapes |
| `env.compile.backend` | `"inductor"` | Compilation backend |

## Model

### model.names

Available model types (unneeded modalities auto-removed):

| Type | Description |
|------|-------------|
| `"hf_text"` | Huggingface pretrained text models |
| `"timm_image"` | TIMM pretrained image models |
| `"clip"` | Pretrained CLIP models |
| `"categorical_mlp"` / `"numerical_mlp"` | MLPs for tabular data |
| `"ft_transformer"` | FT-Transformer for tabular data |
| `"fusion_mlp"` / `"fusion_transformer"` | Multi-backbone fusion |
| `"sam"` | Segment Anything Model |

```python
# default
predictor.fit(hyperparameters={"model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"]})

# single modality
predictor.fit(hyperparameters={"model.names": ["hf_text"]})
```

### Text Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hf_text.checkpoint_name` | `"google/electra-base-discriminator"` | Any Huggingface AutoModel checkpoint |
| `model.hf_text.pooling_mode` | `"cls"` | `"cls"` (CLS token) or `"mean"` (average all tokens) |

```python
predictor.fit(hyperparameters={"model.hf_text.checkpoint_name": "roberta-base"})
```

### Other Parameters

**`optim.skip_final_val`** (default: `False`): Skip final validation after training stops.

# AutoMM Text, Tabular & Image Model Configuration

## Text Model (hf_text) — Continued

### Tokenization & Text Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hf_text.tokenizer_name` | `"hf_auto"` | Recommended default. Options: `"hf_auto"`, `"bert"`, `"electra"`, `"clip"` |
| `model.hf_text.max_text_len` | `512` | Max token length. Uses `min(this, model_max)`. Set ≤0 for model's max |
| `model.hf_text.insert_sep` | `True` | Insert SEP token between texts from different dataframe columns |
| `model.hf_text.text_segment_num` | `2` | Number of text segments (token type IDs). Uses `min(this, model_default)` |
| `model.hf_text.stochastic_chunk` | `False` | Randomly sample start index for over-long sequences instead of truncating from 0 |

### Text Augmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hf_text.text_aug_detect_length` | `10` | Min token count to allow augmentation |
| `model.hf_text.text_trivial_aug_maxscale` | `0` (off) | Max % of tokens to augment (synonym replacement, word swap/deletion, punctuation insertion) |

```python
# Enable trivial augmentation
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0.1})
```

### Memory Optimization

**`model.hf_text.gradient_checkpointing`** (default: `False`): Reduces memory for gradient computation at cost of speed.

## FT-Transformer Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.ft_transformer.checkpoint_name` | `None` | Local path or URL to pretrained weights |
| `model.ft_transformer.num_blocks` | `3` | Number of transformer blocks |
| `model.ft_transformer.token_dim` | `192` | Token dimension after tokenizers |
| `model.ft_transformer.hidden_size` | `192` | Model embedding dimension |
| `model.ft_transformer.ffn_hidden_size` | `192` | FFN hidden dimension (original Transformer uses 4× hidden_size; here defaults to 1×) |

## Image Model (timm_image)

### Backbone Selection

```python
# default
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "swin_base_patch4_window7_224"})
# alternative
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "vit_base_patch32_224"})
```

### Training Transforms

**`model.timm_image.train_transforms`** — accepts a list of strings or callable pickle-able objects:

Built-in options: `resize_to_square`, `resize_shorter_side`, `center_crop`, `random_resize_crop`, `random_horizontal_flip`, `random_vertical_flip`, `color_jitter`, `affine`, `randaug`, `trivial_augment`

```python
# default
predictor.fit(hyperparameters={"model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"]})

# custom torchvision transforms
predictor.fit(hyperparameters={"model.timm_image.train_transforms": [
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip()
]})
```

# AutoMM Image Models, SAM, MMDetection & Data Configuration

## Image Model (timm_image) — Continued

**`model.timm_image.val_transforms`** (default: `["resize_shorter_side", "center_crop"]`): Accepts strings or callable pickle-able objects, same as train_transforms.

## MMDetection Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.mmdet_image.checkpoint_name` | `"yolov3_mobilenetv2_8xb24-320-300e_coco"` | Use `"yolox_nano/tiny/s/m/l/x"` for AG-compatible YOLOX models |
| `model.mmdet_image.output_bbox_format` | `"xyxy"` | `"xyxy"` (corners) or `"xywh"` (corner + width/height) |
| `model.mmdet_image.frozen_layers` | `[]` | Freeze layers by substring match, e.g., `["backbone", "neck"]` |

```python
predictor = MultiModalPredictor(hyperparameters={"model.mmdet_image.checkpoint_name": "dino-5scale_swin-l_8xb2-36e_coco"})
```

## SAM (Segment Anything Model)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.sam.checkpoint_name` | `"facebook/sam-vit-huge"` | Also: `sam-vit-large`, `sam-vit-base` |
| `model.sam.train_transforms` | `["random_horizontal_flip"]` | Training augmentation |
| `model.sam.img_transforms` | `["resize_to_square"]` | Input image processing |
| `model.sam.gt_transforms` | `["resize_gt_to_square"]` | Ground truth mask processing |
| `model.sam.frozen_layers` | `["mask_decoder.iou_prediction_head", "prompt_encoder"]` | Frozen modules |
| `model.sam.num_mask_tokens` | `1` | Mask proposals from decoder |
| `model.sam.ignore_label` | `255` | Target value ignored in loss/metrics |

## Data Configuration

### Missing Data & Normalization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.image.missing_value_strategy` | `"zero"` | `"zero"` (replace with zero image) or `"skip"` (skip sample) |
| `data.text.normalize_text` | `False` | Fix encoding problems via encode/decode normalization |

### Modality Conversion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.categorical.convert_to_text` | `False` | If `True`, disables categorical models (MLP/transformer) |
| `data.numerical.convert_to_text` | `False` | If `True`, disables numerical models (MLP/transformer) |

### Numerical Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.numerical.scaler_with_mean` | `True` | Center features before scaling (excludes labels) |
| `data.numerical.scaler_with_std` | `True` | Scale features to unit variance (excludes labels) |
| `data.label.numerical_preprocessing` | `"standardscaler"` | Regression label scaling: `"standardscaler"` or `"minmaxscaler"` |

### Classification

**`data.pos_label`** (default: `None`): **Required** for proper use of `roc_auc`, `average_precision`, and `f1` metrics in binary classification.

```python
predictor.fit(hyperparameters={"data.pos_label": "changed"})
```

### Feature Pooling

**`data.column_features_pooling_mode`**: Aggregates multi-column features (for `few_shot_classification` only). Options: `"concat"` or `"mean"`.

# AutoMM Mixup/Cutmix & Knowledge Distillation


...(truncated)