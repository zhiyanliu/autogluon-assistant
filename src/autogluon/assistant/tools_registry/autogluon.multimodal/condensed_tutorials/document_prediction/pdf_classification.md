# Condensed: Get the PDF document dataset

Summary: This tutorial demonstrates PDF document classification using AutoGluon's `MultiModalPredictor` with the LayoutLM transformer model. It covers dataset preparation (loading zips, path expansion via `path_expander`), training a classifier that automatically handles PDF processing and OCR, and configuring hyperparameters like `model.document_transformer.checkpoint_name` and `optim.top_k_average_method`. Key functionalities include `fit()`, `evaluate()`, `predict()`, `predict_proba()`, and `extract_embedding()` for document feature extraction. Useful for building PDF classifiers, document-level inference, and extracting document embeddings with minimal code using AutoGluon's multimodal framework.

*This is a condensed version that preserves essential implementation details and context.*

# PDF Document Classification with AutoGluon MultiModalPredictor

## Dataset Setup

Dataset: 40 PDFs (20 resumes, 20 historical documents) split 80/20 for train/test.

```python
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)
```

**Important:** Expand document paths so `MultiModalPredictor` can locate files:

```python
from autogluon.multimodal.utils.misc import path_expander

DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

## Training the Classifier

AutoMM automatically handles PDF detection, conversion, and text recognition (OCR).

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

**Key parameters:** Uses `layoutlm-base-uncased` (document-aware transformer); `top_k_average_method="best"` selects the best checkpoint.

## Evaluation & Inference

```python
# Evaluate
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])

# Predict single document
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})

# Predict probabilities
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})

# Extract embeddings (N-dimensional, model-dependent)
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(feature[0].shape)
```