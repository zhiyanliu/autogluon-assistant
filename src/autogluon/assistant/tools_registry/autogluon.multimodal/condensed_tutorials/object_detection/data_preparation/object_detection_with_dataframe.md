# Condensed: AutoMM Detection - Object detection data formats

Summary: This tutorial covers AutoGluon AutoMM Detection data format handling, including COCO JSON structure (categories, images, annotations with `[x, y, width, height]` bboxes) and DataFrame format (with `image`, `rois` as `[x1, y1, x2, y2, class_label]`, and `label` columns). It demonstrates format conversion using `from_coco()` and `object_detection_df_to_coco()`, and shows how to configure `MultiModalPredictor` for object detection with YOLOv3/mmdet checkpoints, including key hyperparameters (`optim.lr`, `max_epochs`, `per_gpu_batch_size`). Essential for tasks involving detection data preparation, format conversion, and training/evaluation pipeline setup.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Detection - Data Formats

AutoMM Detection supports **COCO format** and **DataFrame format**.

## COCO Format

Requires a `.json` file with three key sections:

```python
data = {
    "categories": [
        {"supercategory": "none", "id": 1, "name": "person"},
        # ...
    ],
    "images": [
        {"file_name": "<imagename0>.<ext>", "height": 427, "width": 640, "id": 1},
        # ...
    ],
    "annotations": [
        {
            'area': 33453,
            'iscrowd': 0,
            'bbox': [181, 133, 177, 189],  # [x, y, width, height]
            'category_id': 8,  # matches category "id", not "name"
            'ignore': 0,
            'segmentation': [],
            'image_id': 1617,
            'id': 1
        },
        # ...
    ],
    "type": "instances"
}
```

## DataFrame Format

Requires 3 columns:
- **`image`**: path to image file
- **`rois`**: list of arrays `[x1, y1, x2, y2, class_label]`
- **`label`**: copy of `rois`

## Format Conversion

```python
from autogluon.multimodal.utils.object_detection import from_coco, object_detection_df_to_coco

# COCO → DataFrame
train_df = from_coco(train_path)

# DataFrame → COCO (optionally save to file)
train_coco = object_detection_df_to_coco(train_df, save_path="./df_converted_to_coco.json")

# Loading saved COCO: ensure <root>/<file_name> is a valid image path
train_df_from_saved_coco = from_coco("./df_converted_to_coco.json", root="./")
```

## Training & Evaluation

```bash
pip install autogluon.multimodal
mim install mmcv
pip install "mmdet==3.1.0"
```

```python
from autogluon.multimodal import MultiModalPredictor

checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"

predictor_df = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": -1,  # use all GPUs
    },
    problem_type="object_detection",
    sample_data_path=train_df,  # needed to get num_classes
    path=model_path,
)

predictor_df.fit(
    train_df,
    hyperparameters={
        "optim.lr": 2e-4,          # detection head uses 100x lr
        "optim.max_epochs": 30,
        "env.per_gpu_batch_size": 32,  # decrease for large models
    },
)

# Evaluate
test_df = from_coco(test_path)
predictor_df.evaluate(test_df)
```

**Key notes:**
- `sample_data_path` must be provided to infer `num_classes`
- Decrease `per_gpu_batch_size` for larger models
- Two-stage learning rate: detection head gets 100× the base `optim.lr`