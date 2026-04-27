# Condensed: Get a Document Dataset

Summary: This tutorial demonstrates scanned document classification using AutoGluon's MultiModalPredictor, which automatically performs OCR and leverages text, layout, and visual features. It covers dataset loading with `load_zip`, path expansion via `path_expander` for document image paths, training with configurable document transformer models (layoutlmv3, layoutlmv2, layoutlm, layoutxlm, bert, deberta) through `model.document_transformer.checkpoint_name`, evaluation with accuracy metrics, single-document prediction with `predict`/`predict_proba`, and embedding extraction via `extract_embedding`. Useful for building document classifiers on scanned/image-based documents with minimal code using AutoGluon's automated multimodal pipeline.

*This is a condensed version that preserves essential implementation details and context.*

# Scanned Document Classification with AutoMM

## Dataset Setup

Sample of RVL-CDIP: ~100 documents in 3 categories — budget (0), email (1), form (2).

```python
!pip install autogluon.multimodal
```

```python
import warnings
warnings.filterwarnings('ignore')
import os, pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial_doc_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, "rvl_cdip_sample")
rvl_cdip_data = pd.read_csv(f"{dataset_path}/rvl_cdip_train_data.csv")
train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
test_data = rvl_cdip_data.drop(train_data.index)
```

**Important:** Expand document paths to absolute paths for training:

```python
from autogluon.multimodal.utils.misc import path_expander
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

## Training the Classifier

AutoMM automatically performs OCR, then leverages recognized text, layout information, and visual features for classification. Customize the foundation model via `model.document_transformer.checkpoint_name` — supports **layoutlmv3**, **layoutlmv2**, **layoutlm-base**, **layoutxlm**, and pure text models like **bert**, **deberta**.

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
        "optim.top_k_average_method": "best",
    },
    time_limit=120,
)
```

## Evaluation & Inference

```python
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])
```

**Predict single document:**
```python
doc_path = test_data.iloc[1][DOC_PATH_COL]
predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
proba = predictor.predict_proba({DOC_PATH_COL: [doc_path]})
```

## Extract Embeddings

Returns N-dimensional document features (N depends on model):

```python
feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})
print(feature[0].shape)
```