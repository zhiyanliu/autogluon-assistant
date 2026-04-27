# Condensed: 2. Formatting ``*_labels.json``

Summary: This tutorial covers COCO format dataset structure and JSON annotation schema for object detection using AutoGluon's AutoMM. It details the required directory layout (images/ and annotations/ folders), the complete `*_labels.json` specification including images, annotations (with bbox, segmentation, category_id), and categories fields, noting which fields are optional vs. required for training, evaluation, and prediction. It provides VOC-to-COCO conversion via `autogluon.multimodal.cli.voc2coco` with customizable train/val/test split ratios, and mentions FiftyOne for converting CVAT, YOLO, and KITTI formats to COCO.

*This is a condensed version that preserves essential implementation details and context.*

# COCO Format Dataset Structure & Conversion

## Directory Structure
```
<dataset_dir>/
    images/
        <imagename0>.<ext>
        ...
    annotations/
        train_labels.json
        val_labels.json
        test_labels.json
```

## `*_labels.json` Format

```javascript
{
    "info": info,              // optional for AutoMM
    "licenses": [license],     // optional for AutoMM
    "images": [image],         // required (training, eval, prediction)
    "annotations": [annotation], // required (training, eval only)
    "categories": [category]   // required (training, eval only)
}

image = {
    "id": int, "width": int, "height": int,
    "file_name": str, "license": int, "date_captured": datetime
}

category = { "id": int, "name": str, "supercategory": str }

annotation = {
    "id": int, "image_id": int, "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float, "bbox": [x,y,width,height], "iscrowd": 0|1
}
```

## Converting VOC to COCO Format

Expected VOC structure:
```
<path_to_VOCdevkit>/
    VOC2007/
        Annotations/
        ImageSets/
        JPEGImages/
        labels.txt
```

**Conversion commands:**
```python
# Custom train/val/test ratio (test_ratio = 1 - train_ratio - val_ratio)
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>
# Use dataset-provided splits
python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
```

## Other Format Conversions

Any data conforming to COCO format works with AutoMM. Third-party tools like [FiftyOne](https://github.com/voxel51/fiftyone) can convert CVAT, YOLO, KITTI, etc. to COCO format.