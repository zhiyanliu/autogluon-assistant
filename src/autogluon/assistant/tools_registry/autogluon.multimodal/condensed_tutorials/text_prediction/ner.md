# Condensed: Training

Summary: This tutorial covers building a Named Entity Recognition (NER) pipeline using AutoGluon's `MultiModalPredictor` with `problem_type="ner"`. It details the required JSON annotation format (`entity_group`, `start`, `end` keys), optional BIO tagging, and data loading via `load_pd`. Key tasks include training with configurable backbone models (e.g., `google/electra-small-discriminator`) and `time_limit`, evaluation using seqeval metrics (`overall_f1`, `overall_precision`, `overall_recall`, plus per-entity metrics), prediction with `predict()`/`predict_proba()`, visualization via `visualize_ner`, and model reloading with continuous training using `MultiModalPredictor.load()`.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon NER Tutorial (Condensed)

## Setup & Data Format

```python
!pip install autogluon.multimodal
```

NER annotations use JSON with **required keys**: `entity_group`, `start` (char-level begin position), `end` (char-level end position):

```python
[{"entity_group": "PERSON", "start": 0, "end": 15},
 {"entity_group": "LOCATION", "start": 28, "end": 35}]
```

**BIO format** is optional — you can use `B-`/`I-` prefixes (e.g., `B-PERSON`, `I-PERSON`). `O` tags are handled automatically.

Visualize annotations:
```python
from autogluon.multimodal.utils import visualize_ner
visualize_ner(sentence, annotation)
```

## Loading Data

```python
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/mit-movies/train_v2.csv')
test_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/mit-movies/test_v2.csv')
```

Dataset has `text_snippet` and `entity_annotations` columns.

## Training

Set `problem_type="ner"`. **Recommended: use longer `time_limit`** (30-60 min) for real applications.

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'google/electra-small-discriminator'},
    time_limit=300,
)
```

## Evaluation

Uses [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval). Metrics: `overall_recall`, `overall_precision`, `overall_f1`, `overall_accuracy`. Use entity group name (e.g., `"actor"`) for per-entity metrics.

```python
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1", "actor"])
```

## Prediction & Visualization

```python
sentence = "Game of Thrones is an American fantasy drama television series created by David Benioff"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

## Prediction Probabilities

```python
predictions = predictor.predict_proba({'text_snippet': [sentence]})
print(predictions[0][0]['probability'])
```

## Reload & Continue Training

```python
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1', 'ACTOR'])
```