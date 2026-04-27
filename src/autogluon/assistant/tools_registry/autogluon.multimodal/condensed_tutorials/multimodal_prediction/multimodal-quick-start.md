# Condensed: Example Data

Summary: This tutorial demonstrates AutoGluon's `MultiModalPredictor` for multimodal classification/regression combining image, text, and tabular data. It covers data preparation (formatting image columns as absolute file paths, handling multi-image records by selecting one), training with `MultiModalPredictor(label=).fit(train_data, time_limit=)`, and inference via `predict()`, `predict_proba()`, and `evaluate(metrics=[])`. Key implementation details include the requirement that image columns contain single file path strings, automatic problem type inference and late-fusion model selection, and time-budgeted training. Useful for building multimodal ML pipelines with minimal configuration using the PetFinder dataset as an example.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModalPredictor Quick Tutorial

## Setup & Data Preparation

```python
!pip install autogluon
```

```python
from autogluon.core.utils.loaders import load_zip
import pandas as pd
import os

# Download PetFinder dataset (binary classification: adoption speed 0=slow, 1=fast)
download_dir = './ag_multimodal_tutorial'
load_zip.unzip('https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip', unzip_dir=download_dir)

dataset_path = f'{download_dir}/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

**Key requirement:** Image columns must contain a path to a **single image file** as a string. For multi-image records, select one and expand to absolute paths:

```python
image_col = 'Images'
# Keep only first image per record
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Convert to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120  # seconds; more time = better performance
)
```

**Under the hood:** Automatically infers problem type (classification/regression), detects feature modalities (image, text, tabular), selects models, and trains them. Multiple backbones get a **late-fusion model** (MLP or transformer) on top.

## Prediction & Evaluation

```python
# Class predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Class probabilities (classification only)
probs = predictor.predict_proba(test_data.drop(columns=label_col))

# Evaluate with specific metrics
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

## Key Takeaways

- **MultiModalPredictor** handles mixed modalities (images, text, tabular) in a single API
- Image columns must be **string paths to single image files**
- `time_limit` controls training budget; even short budgets yield good results
- Supports additional capabilities: embedding extraction, distillation, model fine-tuning, text/image prediction, semantic matching