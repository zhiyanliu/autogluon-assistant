# Condensed: Train your Model

Summary: This tutorial demonstrates how to build a semantic text similarity model using AutoGluon's `MultiModalPredictor` with `problem_type="text_similarity"`. It covers configuring key parameters (`query`, `response`, `label`, `match_label`, `eval_metric`) for sentence-pair matching, training a BERT-based model on SNLI data with `time_limit`, and performing evaluation, prediction, probability estimation, and per-sentence embedding extraction. It helps with coding tasks involving semantic matching, duplicate detection, sentence-pair classification, and extracting sentence embeddings. Important: labels must be binary, and `match_label` should be set according to task-specific semantics.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Text Similarity (Semantic Matching)

## Setup & Data

```python
!pip install autogluon.multimodal
```

```python
from autogluon.core.utils.loaders import load_pd
import pandas as pd

snli_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_train.csv', delimiter="|")
snli_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_test.csv', delimiter="|")
```

## Training

Uses BERT to project sentences into high-dimensional vectors, treating matching as a classification problem (following [sentence transformers](https://www.sbert.net/) design).

**Key parameters:** `query`/`response` specify sentence columns, `label` is the target, and `match_label` indicates which label means semantically equivalent. **Labels must be binary.** Define `match_label` based on your task context (e.g., duplicate vs. not duplicate).

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="text_similarity",
    query="premise",
    response="hypothesis",
    label="label",
    match_label=1,
    eval_metric='auc',
)

predictor.fit(train_data=snli_train, time_limit=180)
```

## Evaluate, Predict & Extract Embeddings

```python
# Evaluate
score = predictor.evaluate(snli_test)

# Predict on new pairs
pred_data = pd.DataFrame.from_dict({
    "premise": ["The teacher gave his speech to an empty room."],
    "hypothesis": ["There was almost nobody when the professor was talking."]
})
predictions = predictor.predict(pred_data)
probabilities = predictor.predict_proba(pred_data)

# Extract embeddings separately for each sentence group
embeddings_1 = predictor.extract_embedding({"premise": ["The teacher gave his speech to an empty room."]})
embeddings_2 = predictor.extract_embedding({"hypothesis": ["There was almost nobody when the professor was talking."]})
```

**References:** [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) | [Customize AutoMM](../advanced_topics/customization.ipynb)