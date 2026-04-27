# Condensed: Dataset

Summary: This tutorial demonstrates image-text matching using AutoGluon's `MultiModalPredictor` with CLIP backbone. It covers initializing a predictor with `problem_type="image_text_similarity"`, configuring `query`/`response` columns, zero-shot evaluation and finetuning with `recall@k` metrics (cutoffs), binary match prediction (`predict`/`predict_proba`), embedding extraction for images and text via `extract_embedding`, and bidirectional semantic search using `semantic_search()`. It helps with tasks including cross-modal retrieval, image-text similarity scoring, CLIP finetuning on custom datasets, and building semantic search systems. Key data preparation includes expanding image paths and constructing relevance-labeled test data with unique query/response sets.

*This is a condensed version that preserves essential implementation details and context.*

# Image-Text Matching with AutoGluon MultiModal

## Setup & Dataset

```python
!pip install autogluon.multimodal
```

Uses **Flickr30K** dataset (31,783 images with descriptive captions). Each image has 5 captions, so image paths are duplicated 5 times to build correspondences.

```python
from autogluon.core.utils.loaders import load_zip
import pandas as pd, os

download_dir = './ag_automm_tutorial_imgtxt'
load_zip.unzip('https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip', unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, 'flickr30k_processed')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col, text_col = "image", "caption"

# Expand relative paths to absolute
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for df in [train_data, val_data, test_data]:
    df[image_col] = df[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

Prepare test data with unique images/texts and relevance labels:

```python
test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
test_data_with_label = test_data.copy()
test_label_col = "relevance"
test_data_with_label[test_label_col] = [1] * len(test_data)
```

## Initialize Predictor

Set `problem_type="image_text_similarity"`. `query` and `response` are interchangeable between image/text columns. Loads pretrained **CLIP** (`openai/clip-vit-base-patch32`).

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
    query=text_col,
    response=image_col,
    problem_type="image_text_similarity",
    eval_metric="recall",
)
```

## Evaluate (Zero-shot) & Finetune

```python
# Evaluate both retrieval directions
txt_to_img_scores = predictor.evaluate(
    data=test_data_with_label, query_data=test_text_data,
    response_data=test_image_data, label=test_label_col, cutoffs=[1, 5, 10])
img_to_txt_scores = predictor.evaluate(
    data=test_data_with_label, query_data=test_image_data,
    response_data=test_text_data, label=test_label_col, cutoffs=[1, 5, 10])
```

> **Note:** Image-to-text `recall@1` upper bound is 20% since each image maps to 5 texts—only one of five can be top-1.

```python
# Finetune
predictor.fit(train_data=train_data, tuning_data=val_data, time_limit=180)
```

Re-evaluate after finetuning using the same `evaluate()` calls — expect **large improvements** over zero-shot.

## Predict & Extract Embeddings

```python
# Match prediction (binary)
pred = predictor.predict(test_data.head(5))

# Match probabilities (2nd column = match probability)
proba = predictor.predict_proba(test_data.head(5))

# Extract embeddings
image_embeddings = predictor.extract_embedding({image_col: test_image_data[image_col][:5].tolist()})
text_embeddings = predictor.extract_embedding({text_col: test_text_data[text_col][:5].tolist()})
```

## Semantic Search

```python
from autogluon.multimodal.utils import semantic_search

# Text → Image search
text_to_image_hits = semantic_search(
    matcher=predictor, query_data=test_text_data.iloc[[3]],
    response_data=test_image_data, top_k=5)

# Image → Text search
image_to_text_hits = semantic_search(
    matcher=predictor, query_data=test_image_data.iloc[[6]],
    response_data=test_text_data, top_k=5)

# Access top result
top_image_id = text_to_image_hits[0][0]['response_id']
top_text_id = image_to_text_hits[0][1]['response_id']
```

**Key concepts:** `problem_type="image_text_similarity"` enables CLIP-based matching; `query`/`response` are symmetric; `cutoffs` control recall@k evaluation; finetuning on domain data significantly boosts retrieval performance.