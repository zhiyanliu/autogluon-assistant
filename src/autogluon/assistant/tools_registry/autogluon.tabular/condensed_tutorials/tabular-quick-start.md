# Condensed: Example Data

Summary: This tutorial demonstrates AutoGluon's TabularPredictor for automated machine learning on tabular data. It covers installing AutoGluon, loading data via TabularDataset (a pandas DataFrame subclass), training models with automatic task detection (classification/regression), feature engineering, model selection, and ensembling using a simple `fit()` call. Key techniques include setting `time_limit` to control training duration, making predictions with `predict()`, evaluating with `evaluate()`, and comparing individual model performance via `leaderboard()`. Useful for generating code that builds end-to-end ML pipelines on tabular data with minimal configuration and no manual hyperparameter tuning.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Tabular Quick Start

AutoGluon automates feature engineering, model selection, training, and ensembling for tabular data with minimal code.

## Setup & Data Loading

```python
!python -m pip install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
```

`TabularDataset` is a pandas `DataFrame` subclass—all DataFrame methods work on it.

```python
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
label = 'signature'
```

AutoGluon auto-detects task type (here, multi-class classification with 18 classes) and corrects data types (e.g., categorical recognition).

## Training

```python
predictor = TabularPredictor(label=label).fit(train_data)
```

No feature engineering or hyperparameter tuning needed. AutoGluon trains multiple models and ensembles them automatically.

**Key parameter:** `fit(..., time_limit=60)` caps training at 60 seconds. Higher limits → better performance; excessively low limits prevent reasonable model ensembling.

## Prediction & Evaluation

```python
test_data = TabularDataset(f'{data_url}test.csv')
y_pred = predictor.predict(test_data.drop(columns=[label]))
predictor.evaluate(test_data, silent=True)
predictor.leaderboard(test_data)  # per-model performance comparison
```

`evaluate()` measures performance on unseen data; `leaderboard()` ranks all individual trained models on test data.