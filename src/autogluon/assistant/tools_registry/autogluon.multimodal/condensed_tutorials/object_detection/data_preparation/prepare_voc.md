# Condensed: Download with Bash Script

Summary: This tutorial covers downloading and preparing Pascal VOC (2007/2012) datasets for object detection using AutoGluon's MultiModal module. It demonstrates CLI commands (`autogluon.multimodal.cli.prepare_detection_dataset`) and bash scripts for dataset download, with options for specifying output paths and downloading VOC2007/VOC2012 separately. It documents the expected VOC directory structure (`VOCdevkit/` with `Annotations`, `ImageSets`, `JPEGImages`). A key recommendation is to use COCO format over VOC for AutoGluon MultiModalPredictor, with VOC support being limited. This helps with tasks involving dataset preparation, format understanding, and detection pipeline setup.

*This is a condensed version that preserves essential implementation details and context.*

## VOC Dataset Preparation

### Download via CLI

```bash
# Full command with output path
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc -o ~/data

# Download VOC2007 and VOC2012 separately
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc07 -o ~/data
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d voc12 -o ~/data
```

### Download via Bash Script

```bash
bash download_voc0712.sh ~/data  # or omit path to extract in current directory
```

Extracted structure: `VOCdevkit/{VOC2007, VOC2012}/`, each containing:
```
Annotations  ImageSets  JPEGImages  SegmentationClass  SegmentationObject
```

## VOC Format

**⚠️ AutoGluon strongly recommends using COCO format instead of VOC.** See [Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for conversion guidance.

VOC format uses `.xml` annotation files. For limited VOC support, provide the dataset root path containing at minimum:
```
Annotations  ImageSets  JPEGImages
```