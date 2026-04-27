# Condensed: Dataset

Summary: This tutorial demonstrates using AutoGluon's `MultiModalPredictor` for multimodal classification combining images, text, and tabular data. It covers data preparation (image path expansion for semicolon-delimited paths), training with `time_limit`, evaluation with metrics like `roc_auc`, prediction (`predict`, `predict_proba`), embedding extraction via `extract_embedding`, and model save/load. Key techniques include automatic problem type detection, modality inference, late-fusion of multiple backbones, and handling multi-image columns. Useful for coding tasks involving multimodal ML pipelines, binary classification, and feature embedding generation with minimal configuration.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Quick Start

## Setup & Data Preparation

```python
!pip install autogluon.multimodal
```

**Dataset**: Simplified PetFinder dataset — binary classification (adoption speed: 0=slow, 1=fast) using images, text descriptions, and tabular features.

```python
import os, numpy as np, pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial'
load_zip.unzip('https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip', unzip_dir=download_dir)

dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

**Image path expansion** (required for training to locate image files):

```python
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])  # first image only
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

AutoMM auto-detects problem type, data modalities, selects models, and applies late-fusion (MLP/transformer) when multiple backbones are used.

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label=label_col)
predictor.fit(train_data=train_data, time_limit=120)
```

## Evaluation, Prediction & Embeddings

```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
predictions = predictor.predict(test_data.drop(columns=label_col))
probas = predictor.predict_proba(test_data.drop(columns=label_col))  # classification only
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))  # returns array of shape (n_samples, embed_dim)
```

> **Note:** `predict_proba()` raises an exception on regression tasks.

## Save and Load

> ⚠️ **Security Warning:** `MultiModalPredictor.load()` uses `pickle` implicitly. **Only load data you trust.**

```python
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-saved_model"
predictor.save(model_path)
loaded_predictor = MultiModalPredictor.load(model_path)
```

## Further Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)