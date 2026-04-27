# Condensed: If You Download From Kaggle

Summary: This tutorial covers preparing the Pothole object detection dataset using AutoGluon's CLI tool (`autogluon.multimodal.cli.prepare_detection_dataset`). It demonstrates downloading and organizing detection datasets in COCO format with automatic train/val/test splitting (3:1:1 ratio). Key details include annotation file paths and naming conventions. It warns against using VOC format directly, strongly recommending COCO format for AutoGluon MultiModalPredictor, and references companion tutorials for VOC-to-COCO conversion. Useful for tasks involving dataset preparation for object detection pipelines, CLI-based dataset management, and understanding COCO annotation structure in AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

## Pothole Dataset Preparation

**Download using AutoGluon CLI:**

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d pothole -o ~/data
```

Dataset is in **COCO format**, split train/val/test at **3:1:1** ratio. Annotation files:

```
pothole/Annotations/usersplit_train_cocoformat.json
pothole/Annotations/usersplit_val_cocoformat.json
pothole/Annotations/usersplit_test_cocoformat.json
```

> **⚠️ Important:** If downloading from Kaggle, the original dataset is VOC format and unsplit. **AutoMM strongly recommends COCO format.** See [Prepare COCO2017](prepare_coco17.ipynb) and [Convert to COCO Format](convert_data_to_coco_format.ipynb) for VOC→COCO conversion guidance.