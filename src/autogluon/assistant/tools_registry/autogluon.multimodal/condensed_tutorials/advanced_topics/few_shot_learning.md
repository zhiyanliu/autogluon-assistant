# Condensed: Few Shot Text Classification

Summary: This tutorial demonstrates few-shot classification using AutoGluon's `MultiModalPredictor` with `problem_type="few_shot_classification"` for low-data scenarios (8-10 samples per class). It covers both text classification (MLDoc dataset) and image classification (Stanford Cars, 196 classes), showing how to load data into DataFrames, initialize the predictor with key parameters (`problem_type`, `label`, `eval_metric`), train with `.fit()`, and evaluate with `.evaluate()` using metrics like accuracy and F1-macro. Image tasks require full file paths in the DataFrame. The few-shot problem type significantly outperforms default classification when training data is limited.

*This is a condensed version that preserves essential implementation details and context.*

# Few Shot Classification with AutoGluon MultiModalPredictor

## Key Concept
Use `problem_type="few_shot_classification"` for significantly better performance on low-data scenarios (e.g., 8-10 samples per class) compared to default `classification`.

## Text Classification (10-shot)

```python
import pandas as pd
import os
from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/nlp_datasets/MLDoc-10shot-en.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)
train_df = pd.read_csv(f"{download_dir}/train.csv", names=["label", "text"])
test_df = pd.read_csv(f"{download_dir}/test.csv", names=["label", "text"])
```

```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_text = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="label",
    eval_metric="acc",
)
predictor_fs_text.fit(train_df)
scores = predictor_fs_text.evaluate(test_df, metrics=["acc", "f1_macro"])
```

## Image Classification (8-shot)

Dataset: Stanford Cars (196 classes, 8 samples/class). DataFrame requires an image path column with full paths to images.

```python
train_df["ImageID"] = download_dir + train_df["ImageID"].astype(str)
test_df["ImageID"] = download_dir + test_df["ImageID"].astype(str)
# Drop non-essential columns (bounding box, metadata)
train_df = train_df_raw.drop(columns=["Source","Confidence","XMin","XMax","YMin","YMax",
    "IsOccluded","IsTruncated","IsGroupOf","IsDepiction","IsInside"])
```

```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_fs_image.fit(train_df)
scores = predictor_fs_image.evaluate(test_df, metrics=["acc", "f1_macro"])
```

## Critical Parameters

| Parameter | Description |
|-----------|-------------|
| `problem_type` | `"few_shot_classification"` for low-data; outperforms default `"classification"` |
| `label` | Column name containing class labels |
| `eval_metric` | Evaluation metric (e.g., `"acc"`) |

**Key takeaway:** `few_shot_classification` substantially outperforms default `classification` in both text and image tasks when training data is limited.