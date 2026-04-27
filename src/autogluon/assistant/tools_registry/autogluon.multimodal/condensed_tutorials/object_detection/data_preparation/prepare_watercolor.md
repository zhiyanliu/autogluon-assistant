# Condensed: Download with Bash Script

Summary: This tutorial covers downloading and preparing the Watercolor object detection dataset for use with AutoGluon's MultiModalPredictor. It demonstrates two download methods: AutoGluon's Python CLI (`prepare_detection_dataset` module) and a bash script, both supporting custom output paths. The dataset uses VOC format (Annotations, ImageSets, JPEGImages directories), though AutoGluon strongly recommends converting to COCO format for full support. This tutorial helps with tasks involving detection dataset preparation, VOC-to-COCO format conversion awareness, and AutoGluon MultiModal detection pipeline setup.

*This is a condensed version that preserves essential implementation details and context.*

## Watercolor Dataset Setup

### Download Options

**Python CLI:**
```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d watercolor -o ~/data
```

**Bash:**
```
bash download_watercolor.sh ~/data
```

Extracted folder `watercolor` contains: `Annotations  ImageSets  JPEGImages`

## Dataset Format

Watercolor uses VOC format. **⚠️ AutoGluon strongly recommends COCO format instead.** See [Prepare COCO2017](prepare_coco17.ipynb) and [Convert to COCO Format](convert_data_to_coco_format.ipynb) for conversion guidance.

VOC format has limited support — input must be the dataset root containing:
```
Annotations  ImageSets  JPEGImages
```