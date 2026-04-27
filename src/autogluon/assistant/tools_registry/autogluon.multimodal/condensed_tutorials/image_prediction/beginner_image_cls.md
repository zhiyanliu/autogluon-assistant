# Condensed: Use AutoMM to Fit Models

Summary: This tutorial demonstrates image classification using AutoGluon's `MultiModalPredictor`, covering the full pipeline: loading image datasets (supporting both file paths and bytearrays interchangeably), training with `predictor.fit()` using `time_limit` control, evaluating with `predictor.evaluate()`, predicting labels/probabilities via `predict()`/`predict_proba()`, extracting image embeddings (512–2048 dim) with `extract_embedding()`, and saving/loading models. It helps with coding tasks involving quick image classifier training, inference on single images using dict input format (`{'image': [path]}`), and feature extraction. Key note: path-trained and bytearray inputs are interchangeable across all methods.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Image Classification

## Setup & Data Loading

```python
!pip install autogluon.multimodal
```

```python
from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = './ag_automm_tutorial_imgcls'
train_data_path, test_data_path = shopee_dataset(download_dir)
```

Dataset: 800 rows, 2 columns (**image** — absolute paths, **label** — target). Supports both image paths and bytearrays:

```python
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

## Training

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(train_data=train_data_path, time_limit=30)
```

- **label**: target column name
- **path**: model save directory
- **time_limit**: training budget in seconds

## Evaluate, Predict & Extract Embeddings

```python
# Evaluate (works with both path and bytearray data)
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])

# Predict single image
predictions = predictor.predict({'image': [image_path]})
proba = predictor.predict_proba({'image': [image_path]})

# Extract embeddings (512–2048 dim vector depending on model)
feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)
```

**Key detail**: Path-trained models work with bytearray inputs and vice versa — all predict/evaluate/extract_embedding methods accept either format interchangeably.

## Save and Load

Model auto-saves after `fit()`. Reload with:

```python
loaded_predictor = MultiModalPredictor.load(model_path)
```

> ⚠️ **Warning**: `MultiModalPredictor.load()` uses `pickle` implicitly. **Never load data from untrusted sources** — malicious pickle data can execute arbitrary code during unpickling.

## Further Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)