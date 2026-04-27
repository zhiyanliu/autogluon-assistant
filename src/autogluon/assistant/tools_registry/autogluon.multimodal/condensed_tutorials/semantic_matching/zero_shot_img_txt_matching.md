# Condensed: Extract Embeddings

Summary: This tutorial demonstrates AutoGluon's `MultiModalPredictor` for image-text similarity tasks using `problem_type="image_text_similarity"`. It covers extracting image and text embeddings via `extract_embedding()`, performing bidirectional semantic search (text→image and image→text retrieval) using the `semantic_search` utility with `top_k` and swappable query/response embeddings, and predicting image-text pair matching with binary predictions and probabilities. Key implementation details include adding batch dimensions with `[None,]` for single queries, reinitializing the predictor with arbitrary `query`/`response` column names for pair matching, and accessing results via `hits[0][0]["response_id"]`.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Image-Text Similarity: Embeddings & Retrieval

## Setup & Data

```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import download, semantic_search

texts = [
    "A cheetah chases prey on across a field.",
    "A man is eating a piece of bread.",
    # ... more texts
    "There is a carriage in the image.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
]
image_paths = [download(url) for url in urls]
```

## Extract Embeddings

Initialize with `problem_type="image_text_similarity"`. Image/text data routes through corresponding encoders automatically.

```python
predictor = MultiModalPredictor(problem_type="image_text_similarity")
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
```

## Image Retrieval (Text Query) / Text Retrieval (Image Query)

Use `semantic_search` with cosine similarity. Swap `query_embeddings`/`response_embeddings` to switch retrieval direction. **Note:** use `[None,]` to add batch dimension to single queries.

```python
# Text → Image retrieval
hits = semantic_search(
    matcher=predictor,
    query_embeddings=text_embeddings[6][None,],
    response_embeddings=image_embeddings,
    top_k=5,
)
# Access result: image_paths[hits[0][0]["response_id"]]

# Image → Text retrieval (swap query/response)
hits = semantic_search(
    matcher=predictor,
    query_embeddings=image_embeddings[4][None,],
    response_embeddings=text_embeddings,
    top_k=5,
)
# Access result: texts[hits[0][0]["response_id"]]
```

## Predict Image-Text Pair Matching

Requires reinitializing predictor with `query` and `response` column names:

```python
predictor = MultiModalPredictor(
    query="abc",
    response="xyz",
    problem_type="image_text_similarity",
)

# Binary match prediction
pred = predictor.predict({"abc": [image_paths[4]], "xyz": [texts[3]]})

# Match probabilities (for custom thresholds)
proba = predictor.predict_proba({"abc": [image_paths[4]], "xyz": [texts[3]]})
```

**Key concepts:** `query`/`response` names are arbitrary but must match the keys in the prediction dict. This enables both `predict` (binary) and `predict_proba` (probability) outputs.