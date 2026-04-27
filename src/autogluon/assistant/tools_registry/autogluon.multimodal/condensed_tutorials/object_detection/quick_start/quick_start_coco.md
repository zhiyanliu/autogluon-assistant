# Condensed: Update package tools and install build dependencies

Summary: This tutorial covers AutoGluon's `MultiModalPredictor` for object detection, including end-to-end workflows: installation (with MMCV/MMDet compatibility warnings for CUDA 12.4+PyTorch 2.5), loading COCO-format datasets, selecting model presets (`medium_quality`/YOLOX, `high_quality`/DINO-ResNet50, `best_quality`/DINO-SwinL), finetuning with two-stage learning rate strategy, evaluation (mAP/mAP50), and inference. It demonstrates prediction on COCO JSON files or image lists, saving results as CSV or COCO JSON, loading saved predictors, and visualizing detections with `ObjectDetectionVisualizer` using confidence thresholds. Useful for implementing object detection pipelines with minimal code.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Object Detection Quick Start

## Installation

**Warning:** MMDet is no longer maintained; only compatible with MMCV 2.1.0. Use CUDA 12.4 + PyTorch 2.5. Restart Jupyter kernel after installation.

```bash
pip install autogluon.multimodal
pip install -U pip setuptools wheel
sudo apt-get install -y ninja-build gcc g++
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"
```

## Setup & Data

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_zip
import os, time

zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"
load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

Dataset requires COCO-format JSON annotation files per split.

## Model Presets

| Preset | Model | Trade-off |
|--------|-------|-----------|
| `medium_quality` | YOLOX-large | Fast finetuning/inference, easy deployment |
| `high_quality` | DINO-Resnet50 | Better accuracy, slower |
| `best_quality` | DINO-SwinL | Best accuracy, highest GPU memory |

## Training

Key: Two-stage learning rate strategy — head layers use **100x higher LR**, accelerating convergence on small datasets.

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",
    path=model_path,  # optional; defaults to AutogluonModels/<timestamp>
)
predictor.fit(train_path)
```

## Evaluation

Returns mAP (COCO standard) and mAP50 (VOC standard).

```python
predictor.evaluate(test_path)
```

Load a saved predictor:
```python
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)
```

## Inference

```python
pred = predictor.predict(test_path)
```

Returns a DataFrame with columns:
- **`image`**: image path
- **`bboxes`**: list of `{"class": str, "bbox": [x1, y1, x2, y2], "score": float}`

Save results:
```python
pred = predictor.predict(test_path, save_results=True, as_coco=True, result_save_path="./results.json")
```

### Custom Input Formats

```python
# From COCO JSON
pred = predictor.predict("input_data_for_demo/demo_annotation.json")

# From image list
pred = predictor.predict(["path/to/image.jpg"])
```

## Visualization

```bash
pip install opencv-python
```

```python
from autogluon.multimodal.utils import ObjectDetectionVisualizer
from PIL import Image

conf_threshold = 0.4
image_result = pred.iloc[30]
visualizer = ObjectDetectionVisualizer(image_result.image)
out = visualizer.draw_instance_predictions(image_result, conf_threshold=conf_threshold)
Image.fromarray(out.get_image(), 'RGB')
```