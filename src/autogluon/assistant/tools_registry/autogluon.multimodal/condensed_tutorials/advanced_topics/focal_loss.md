# Condensed: Create Dataset

Summary: This tutorial demonstrates how to configure focal loss in AutoGluon's MultiModalPredictor for imbalanced image classification. It covers creating artificially imbalanced datasets via downsampling, computing normalized inverse-frequency per-class weights for the alpha parameter, and setting focal loss hyperparameters (`optim.loss_func`, `optim.focal_loss.alpha`, `optim.focal_loss.gamma`, `optim.focal_loss.reduction`) within `predictor.fit()`. It helps with tasks involving imbalanced multiclass image classification using Swin Transformer models, comparing focal loss vs. standard loss baselines, and properly configuring class-balanced weights. Key constraint: alpha list length must match the number of classes.

*This is a condensed version that preserves essential implementation details and context.*

# Focal Loss for Imbalanced Image Classification with AutoGluon

## Setup & Dataset

```python
!pip install autogluon.multimodal
from autogluon.multimodal.utils.misc import shopee_dataset

download_dir = "./ag_automm_tutorial_imgcls_focalloss"
train_data, test_data = shopee_dataset(download_dir)
```

## Creating Imbalanced Data

Downsample training data (each class gets 1/3 of the previous class's samples):

```python
import numpy as np, pandas as pd

ds = 1
imbalanced_train_data = []
for lb in range(4):
    class_data = train_data[train_data.label == lb]
    sample_index = np.random.choice(np.arange(len(class_data)), size=int(len(class_data) * ds), replace=False)
    ds /= 3
    imbalanced_train_data.append(class_data.iloc[sample_index])
imbalanced_train_data = pd.concat(imbalanced_train_data)

# Compute per-class weights (inverse of class frequency, normalized)
weights = []
for lb in range(4):
    class_data = imbalanced_train_data[imbalanced_train_data.label == lb]
    weights.append(1 / (class_data.shape[0] / imbalanced_train_data.shape[0]))
weights = list(np.array(weights) / np.sum(weights))
```

## Training with Focal Loss

**Key parameters:**
- **`optim.focal_loss.alpha`** — per-class weight list; length **must** match number of classes. Use inverse of class sample percentage.
- **`optim.focal_loss.gamma`** — controls focus on hard samples (higher = more focus)
- **`optim.focal_loss.reduction`** — `"mean"` or `"sum"`

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)
predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": weights,
        "optim.focal_loss.gamma": 1.0,
        "optim.focal_loss.reduction": "sum",
        "optim.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)
predictor.evaluate(test_data, metrics=["acc"])
```

## Baseline (Without Focal Loss)

```python
predictor2 = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)
predictor2.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optim.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)
predictor2.evaluate(test_data, metrics=["acc"])
```

**Key takeaway:** Focal loss significantly improves performance on imbalanced datasets. When data is imbalanced, try focal loss with properly computed per-class alpha weights.

**Reference:** Lin et al., *Focal Loss for Dense Object Detection* (2017) — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)