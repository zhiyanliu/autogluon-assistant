# Condensed: Download with Bash Script

Summary: This tutorial covers downloading and preparing the COCO2017 dataset for use with AutoGluon's MultiModalPredictor for object detection tasks. It provides CLI commands (Python and Bash) to download and extract the dataset to a specified directory, producing the standard COCO folder structure (annotations, train2017, val2017, test2017, unlabeled2017). It emphasizes that AutoGluon strongly recommends COCO JSON format for detection data and references companion tutorials for converting other formats (especially VOC) to COCO format. Useful for tasks involving dataset preparation, object detection pipeline setup, and data format conversion in AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

## Downloading COCO2017 Dataset

**Via Python CLI:**
```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d coco17 -o ~/data
```

**Via Bash:**
```
bash download_coco17.sh ~/data
```

Omitting the output path extracts to `./coco17`. The resulting folder structure:
```
annotations  test2017  train2017  unlabeled2017  val2017
```

## COCO Format

AutoGluon MultiModalPredictor **strongly recommends** using COCO format (`.json`) for detection data. See [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) and [VOC to COCO conversion](voc_to_coco.ipynb) for creating COCO-format datasets from scratch or converting from VOC format.