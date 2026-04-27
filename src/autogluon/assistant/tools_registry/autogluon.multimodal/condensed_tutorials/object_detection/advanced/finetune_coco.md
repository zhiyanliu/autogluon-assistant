# Condensed: Update package tools and install build dependencies

Summary: This tutorial demonstrates object detection using AutoGluon's MultiModalPredictor with YOLOX on COCO-format datasets. It covers setup with mmcv/mmdet/mmengine dependencies, loading COCO-format annotation data, creating a predictor with `problem_type="object_detection"` and configurable checkpoints, finetuning with two-stage learning rate (100x on detection head), batch size, early stopping, and validation interval parameters. It also shows simplified training via quality presets (`"medium_quality"`, `"high_quality"`, `"best_quality"`), evaluation returning mAP/mAP50 metrics, prediction, and visualization with confidence thresholding using `visualize_detection`. Useful for implementing end-to-end object detection finetuning pipelines.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Object Detection with YOLOX (Condensed)

## Setup

```python
!pip install autogluon.multimodal
!pip install -U pip setuptools wheel
!sudo apt-get install -y ninja-build gcc g++
!python3 -m mim install "mmcv==2.1.0"
!python3 -m pip install "mmdet==3.2.0"
!python3 -m pip install "mmengine>=0.10.6"
# Colab alternative for MMCV:
# pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

## Data Preparation (COCO Format)

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_zip
import os

zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
download_dir = "./pothole"
load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "pothole")
train_path = os.path.join(data_dir, "Annotations", "usersplit_train_cocoformat.json")
val_path = os.path.join(data_dir, "Annotations", "usersplit_val_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "usersplit_test_cocoformat.json")
```

Input is the COCO-format JSON annotation file for each split.

## Model Creation

```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "yolox_s",
        "env.num_gpus": 1,
    },
    problem_type="object_detection",
    sample_data_path=train_path,  # used to infer dataset categories; any split works
)
```

**Key:** Larger models (change `checkpoint_name`) improve performance but require adjusting `lr` and `per_gpu_batch_size`. Alternatively, use presets: `"medium_quality"`, `"high_quality"`, or `"best_quality"`.

## Training

```python
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optim.lr": 1e-4,                    # two-stage LR: detection head gets 100x
        "env.per_gpu_batch_size": 16,         # adjust based on GPU memory
        "optim.max_epochs": 30,
        "optim.val_check_interval": 1.0,      # validate once per epoch
        "optim.check_val_every_n_epoch": 3,   # validate every 3 epochs
        "optim.patience": 3,                  # early stop after 3 non-improving validations
    },
)
```

**Best practice:** Two-stage learning rate (high LR on head only) improves convergence and performance, especially on small datasets.

## Simplified Approach with Presets

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",
)
predictor.fit(train_path, tuning_data=val_path)
predictor.evaluate(test_path)
```

## Evaluation & Prediction

```python
predictor.evaluate(test_path)  # Returns mAP (COCO standard) and mAP50 (VOC standard)
pred = predictor.predict(test_path)
```

## Visualization

```python
from autogluon.multimodal.utils import visualize_detection
from PIL import Image
from IPython.display import display

visualized = visualize_detection(
    pred=pred[12:13],
    detection_classes=predictor.classes,
    conf_threshold=0.25,
    visualization_result_dir="./",
)
img = Image.fromarray(visualized[0][:, :, ::-1], 'RGB')
display(img)
```

**Performance note:** This fast finetune achieves good mAP in minutes. For higher performance (e.g., VFNet reaching `mAP=0.450, mAP50=0.718`), see the high-performance finetuning guide.