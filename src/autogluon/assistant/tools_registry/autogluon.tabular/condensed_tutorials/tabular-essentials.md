# Condensed: Training

Summary: This tutorial covers AutoGluon's TabularPredictor for automated tabular ML, including classification and regression. It demonstrates data loading (CSV/Parquet), training with `fit()`, prediction (`predict`/`predict_proba`), evaluation (`evaluate`/`leaderboard`), model saving/loading, and feature importance analysis. Key configurations include `presets` (`medium`, `good`, `high`, `best`, `extreme`), `eval_metric` (e.g., `roc_auc`, `f1`, `mean_absolute_error`), and `time_limit`. It covers strategies for maximizing accuracy: using `presets='best'`, avoiding manual preprocessing/HPO, and letting AutoGluon handle ensembling, feature engineering, and data splitting. Useful for implementing end-to-end AutoML pipelines with minimal code.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Tabular Quick Start

## Setup & Data Loading

```python
!pip install autogluon.tabular[all]
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=500, random_state=0)  # subsample for demo
label = 'class'
```

**Best Practice:** Don't preprocess data (no imputation, no one-hot-encoding) — AutoGluon handles this automatically.

## Training & Prediction

```python
predictor = TabularPredictor(label=label).fit(train_data)

test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_pred = predictor.predict(test_data)
y_pred_proba = predictor.predict_proba(test_data)
```

## Evaluation

```python
predictor.evaluate(test_data)       # Overall metrics
predictor.leaderboard(test_data)    # Per-model evaluation
```

## Loading a Saved Predictor

```python
predictor = TabularPredictor.load(predictor.path)
```

> ⚠️ **Warning:** `TabularPredictor.load()` uses `pickle` implicitly. Never load data from untrusted sources — arbitrary code execution is possible during unpickling.

## Minimal Template

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=<variable-name>).fit(train_data=<file-name>)
```

**Note:** This basic `fit()` call is for prototyping. Use `presets` parameter in `fit()` and `eval_metric` in `TabularPredictor()` to maximize performance.

## How `fit()` Works

AutoGluon automatically infers problem type (e.g., binary classification), feature types, evaluation metric, and handles missing data/rescaling. Without explicit validation data, it performs a random train/validation split.

By default, AutoGluon trains **multiple model types** (neural networks, tree ensembles, etc.), automates hyperparameter tuning, and ensembles them. Training parallelizes across threads via [Ray](https://www.ray.io/). Control runtime with `time_limit` in `fit()`.

```python
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
```

### Feature Inspection & Importance

```python
test_data_transform = predictor.transform_features(test_data)  # View internal numeric representation
predictor.feature_importance(test_data)  # Estimate feature importance
```

The `importance` column estimates how much the eval metric drops if the feature is removed. **Negative values suggest removing the feature may improve results.**

### Model Selection

```python
predictor.model_best                          # Best model (default for predict)
predictor.predict(test_data, model='LightGBM')  # Predict with specific model
predictor.model_names()                        # List all trained models
```

**Important:** Default metric (accuracy for binary classification) may not match your application — specify `eval_metric` in `TabularPredictor()` when you know the target metric.

## Presets

Specified via `presets` argument in `.fit()`. Default is `medium`.

| Preset | Quality | Fit Time | Inference Time | Disk |
|:-------|:--------|:---------|:---------------|:-----|
| **extreme** | Far better than best (<30K samples) | 4x+ | 32x+ | 8x+ |
| **best** | SOTA, preferred for serious usage | 16x+ | 32x+ | 16x+ |
| **high** | Better than good; fast inference | 16x+ | 4x | 2x |
| **good** | Stronger than other AutoML frameworks | 16x | 2x | 0.1x |
| **medium** | Competitive; prototyping baseline | 1x | 1x | 1x |

**Recommended workflow:**
1. Start with `medium` to prototype and identify data issues
2. Move to `best` with ≥16x the `time_limit` used in `medium`
3. Try `high`/`good` if specific inference speed or disk constraints exist
4. **`extreme`** (new v1.4): Uses tabular foundation models (TabPFNv2, TabICL, Mitra) + TabM. **Requires GPU.** Install via `pip install autogluon[tabarena]`

**Best Practice:** Always hold out test data that AutoGluon never sees during training to validate performance.

## Maximizing Predictive Performance

**⚠️ Do not use default `fit()` arguments for benchmarking or production accuracy.**

```python
time_limit = 60  # set to longest acceptable wait (seconds)
metric = 'roc_auc'
predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best')
predictor.leaderboard(test_data)
```

**Key strategies:**
- **`presets='best'`**: Enables stacking/bagging ensembles for maximum accuracy. For fast deployment: `presets=['good', 'optimize_for_deployment']`
- **`eval_metric`**: Specify your application's metric — e.g., `'f1'`, `'roc_auc'`, `'log_loss'`, `'mean_absolute_error'`, `'median_absolute_error'`, or a [custom metric](advanced/tabular-custom-metric.ipynb)
- **Include all data in `train_data`**, don't provide `tuning_data` — AutoGluon splits more intelligently on its own
- **Don't specify `hyperparameter_tune_kwargs`** — ensembling outperforms HPO under limited time budgets (use HPO only for single-model deployment)
- **Don't specify `hyperparameters`** — let AutoGluon adaptively select models
- **Set `time_limit` as high as feasible** — performance improves with more time

## Regression

AutoGluon auto-detects regression tasks from numeric labels:

```python
predictor_age = TabularPredictor(label='age', path="agModels-predictAge").fit(train_data, time_limit=60)
predictor_age.evaluate(test_data)
predictor_age.leaderboard(test_data)
```

**Note:** For metrics where lower is better (e.g., RMSE), AutoGluon flips the sign and prints negative values during training (internally assumes higher = better). Override default with `eval_metric='mean_absolute_error'`, etc.

## Data Formats

Supported: **pandas DataFrames**, **CSV**, **Parquet**. Multi-table data must be joined into a single table with rows as independent observations and columns as features.

## Advanced Usage

- [In Depth Tutorial](tabular-indepth.ipynb) — advanced options
- [Deployment Optimization](advanced/tabular-deployment.ipynb)
- [Custom Models](advanced/tabular-custom-model.ipynb)