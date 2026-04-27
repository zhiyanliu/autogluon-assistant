# Condensed: Prepare your Data

Summary: This tutorial demonstrates image-to-image semantic matching using AutoGluon's `MultiModalPredictor` with `problem_type="image_similarity"`. It covers data preparation for image pair datasets (expanding relative paths, setting `match_label`), training a Swin Transformer-based similarity model via `MultiModalPredictor` with `query`/`response`/`label` column configuration, evaluation using ROC-AUC, binary prediction, probability extraction for custom thresholding, and embedding extraction via `extract_embedding()`. Useful for building image similarity/matching systems, product deduplication, or visual search applications with minimal code using AutoGluon's high-level API.

*This is a condensed version that preserves essential implementation details and context.*

# Image-to-Image Semantic Matching with AutoMM

## Setup & Data Preparation

```python
!pip install autogluon.multimodal
```

Uses the **Stanford Online Products (SOP)** dataset — 12 product categories where same-product image pairs are positive (label=1) and different-product pairs are negative (label=0).

```python
download_dir = './ag_automm_tutorial_img2img'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/Stanford_Online_Products.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, 'Stanford_Online_Products')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col_1, image_col_2, label_col, match_label = "Image1", "Image2", "Label", 1
```

**Important:** `match_label` must correspond to the label class indicating a semantic match (here `1`). Set according to your task.

Expand relative image paths to absolute:

```python
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

Uses **Swin Transformer** to project images into high-dimensional vectors and computes **cosine similarity** between them.

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
    problem_type="image_similarity",
    query=image_col_1,
    response=image_col_2,
    label=label_col,
    match_label=match_label,
    eval_metric='auc',
)
predictor.fit(train_data=train_data, time_limit=180)
```

## Evaluation & Inference

```python
# Evaluate (ROC-AUC)
score = predictor.evaluate(test_data)

# Binary predictions (threshold=0.5)
pred = predictor.predict(test_data.head(3))

# Probabilities (for custom thresholding)
proba = predictor.predict_proba(test_data.head(3))
```

## Extract Embeddings

```python
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1][:5].tolist()})
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2][:5].tolist()})
```

**References:** [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) | [Customize AutoMM](../advanced_topics/customization.ipynb)