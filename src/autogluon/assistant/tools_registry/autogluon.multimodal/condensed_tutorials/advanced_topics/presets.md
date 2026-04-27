# Condensed: Dataset

Summary: This tutorial demonstrates using AutoGluon's `MultiModalPredictor` for binary text classification (sentiment analysis) with three quality presets (`medium_quality`, `high_quality`, `best_quality`) that trade off speed vs. performance. It covers loading parquet data via `load_pd`, configuring the predictor with `label`, `eval_metric`, and `presets` parameters, training with `time_limit`, and evaluating with metrics like `roc_auc`. It also shows HPO variants (appending `_hpo`) for automatic tuning of backbone, learning rate, batch size, and optimizer, plus inspecting preset hyperparameters via `get_presets()`. Useful for implementing quick NLP classification pipelines with minimal code.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Presets for Text Classification

## Setup & Data

```python
!pip install autogluon.multimodal
```

Uses subsampled **SST** (Stanford Sentiment Treebank) for binary sentiment classification.

```python
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
train_data = train_data.sample(n=1000, random_state=0)  # subsample for speed
```

## Three Quality Presets

All follow the same pattern—only `presets` and `time_limit` differ:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="<PRESET>")
predictor.fit(train_data=train_data, time_limit=<TIME>)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

| Preset | Model Size | Recommended `time_limit` | Notes |
|---|---|---|---|
| `medium_quality` | Smallest | 20s+ | Fast training/inference |
| `high_quality` | Larger | 20s+ | Balances quality & speed |
| `best_quality` | Largest | 180s+ | Best performance; **needs high-end GPUs with large memory** |

Performance scales: `best_quality` > `high_quality` > `medium_quality`.

## HPO Presets

Append `_hpo` to enable hyperparameter optimization: `medium_quality_hpo`, `high_quality_hpo`, `best_quality_hpo`. Tunes model backbone, batch size, learning rate, max epoch, and optimizer type.

## Inspecting Preset Details

```python
import json
from autogluon.multimodal.utils.presets import get_presets

hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality")
print(json.dumps(hyperparameters, sort_keys=True, indent=4))
print(json.dumps(hyperparameter_tune_kwargs, sort_keys=True, indent=4))
```

## References

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)