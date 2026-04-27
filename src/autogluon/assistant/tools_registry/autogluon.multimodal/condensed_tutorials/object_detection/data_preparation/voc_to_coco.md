# Condensed: Convert Existing Splits

Summary: This tutorial covers converting Pascal VOC format object detection datasets to COCO format using AutoGluon's built-in CLI tool (`autogluon.multimodal.cli.voc2coco`). It explains the expected VOC directory structure (`Annotations`, `ImageSets`, `JPEGImages`) and demonstrates two conversion approaches: converting pre-defined splits (train/val/test.txt) and creating custom train/val/test splits with user-specified ratios via `--train_ratio` and `--val_ratio` flags. This helps with dataset preparation tasks for object detection pipelines, particularly when migrating VOC-format annotations to COCO-format JSON files required by modern detection frameworks.

*This is a condensed version that preserves essential implementation details and context.*

## VOC to COCO Format Conversion

**VOC directory structure:** `Annotations`, `ImageSets`, `JPEGImages` with split files in `ImageSets/Main/` (train.txt, val.txt, test.txt).

### Convert Pre-defined Splits

```
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007
```

Outputs COCO-format JSONs in `Annotations/`: `train_cocoformat.json`, `val_cocoformat.json`, `test_cocoformat.json`.

### Custom Train/Val/Test Splits

No pre-existing split files required. Specify ratios (test ratio is inferred):

```
python3 -m autogluon.multimodal.cli.voc2coco --root_dir ./VOCdevkit/VOC2007 --train_ratio 0.6 --val_ratio 0.2
```

Outputs: `usersplit_train_cocoformat.json`, `usersplit_val_cocoformat.json`, `usersplit_test_cocoformat.json` in `Annotations/`.

**References:** [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) | [Customize AutoMM](../../advanced_topics/customization.ipynb)