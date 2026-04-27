# Condensed: Specifying hyperparameters and tuning them

Summary: This tutorial covers advanced AutoGluon TabularPredictor configuration including hyperparameter tuning with search spaces (`space.Real`, `space.Int`, `space.Categorical`) for GBM and neural network models, stacking/bagging ensembles (`num_bag_folds`, `num_stack_levels`, `auto_stack`), and decision threshold calibration for binary classification metrics. It details inference optimization techniques: `refit_full` for collapsing bagged ensembles, `persist`/`unpersist` for memory management, `infer_limit` constraints, model distillation, alternative ensemble generation via `fit_weighted_ensemble`, and lightweight presets. Also covers model selection via leaderboard, feature importance, saving/loading predictors, memory/disk management strategies, and custom preprocessing options.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Hyperparameter Tuning & Ensembling

## Setup & Data Loading

```python
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common import space

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=1000, random_state=0)
label = 'occupation'
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label]
test_data_nolabel = test_data.drop(columns=[label])
metric = 'accuracy'
```

## Hyperparameter Tuning

> **Not recommended in most cases** — AutoGluon performs best with `presets="best_quality"` without manual HPO.

**Validation data notes:**
- Omit `tuning_data` to let AutoGluon auto-select via stratified sampling; use `holdout_frac` for control
- Only specify `tuning_data` when test distribution differs from training
- **⚠️ Performance on validation data may be over-optimistic** — always evaluate on a separate held-out dataset

### Defining Search Spaces

```python
nn_options = {
    'num_epochs': 10,
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),
}

gbm_options = {
    'num_boost_round': 100,
    'num_leaves': space.Int(lower=26, upper=66, default=36),
}

hyperparameters = {
    'GBM': gbm_options,
    'NN_TORCH': nn_options,  # comment out if errors on Mac OSX
}  # Missing keys = no models of that type trained

hyperparameter_tune_kwargs = {
    'num_trials': 5,
    'scheduler': 'local',
    'searcher': 'auto',
}  # HPO only runs when this is specified

predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    time_limit=2*60,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

### Prediction & Evaluation

```python
y_pred = predictor.predict(test_data_nolabel)
perf = predictor.evaluate(test_data, auxiliary_metrics=False)
results = predictor.fit_summary()  # shows HPO details per model type
```

**Improving results:** Increase `subsample_size`, `num_epochs`, `num_boost_round`, and `time_limit`. Use `verbosity=3` for detailed output.

## Model Ensembling with Stacking/Bagging

Use `num_bag_folds=5-10` and `num_stack_levels=1` to boost performance (increases training time and resource usage).

```python
predictor = TabularPredictor(label='class', eval_metric=metric).fit(
    train_data,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1,
    hyperparameters={'NN_TORCH': {'num_epochs': 2}, 'GBM': {'num_boost_round': 20}},
)
```

**Key practices:**
- **Do not** provide `tuning_data` when stacking/bagging — pass all data as `train_data`
- `num_bag_sets` controls repeated k-fold bagging (higher = less variance, more resources)
- Use `auto_stack` (included in `best_quality` preset) instead of manually setting values

## Auto-Stack with Balanced Accuracy

```python
predictor = TabularPredictor(label=label, eval_metric='balanced_accuracy', path=save_path).fit(
    train_data, auto_stack=True,
    calibrate_decision_threshold=False,  # Disabled for demo
    hyperparameters={'FASTAI': {'num_epochs': 10}, 'GBM': {'num_boost_round': 200}}
)
predictor.leaderboard(test_data)
```

**Key insight:** Stacking/bagging often outperforms hyperparameter-tuning alone. `presets='best_quality'` simply sets `auto_stack=True`. Both techniques can be combined.

## Decision Threshold Calibration

For binary classification, adjusting the decision threshold (default 0.5) can significantly improve metrics like `"f1"` and `"balanced_accuracy"`.

### Calibrate Post-Fit

```python
# Before calibration
scores = predictor.evaluate(test_data)

# Calibrate and apply
calibrated_decision_threshold = predictor.calibrate_decision_threshold()
predictor.set_decision_threshold(calibrated_decision_threshold)

# After calibration
scores_calibrated = predictor.evaluate(test_data)
```

### Calibrate for Specific Metrics

```python
predictor.set_decision_threshold(0.5)  # Reset
for metric_name in ['f1', 'balanced_accuracy', 'mcc']:
    calibrated_decision_threshold = predictor.calibrate_decision_threshold(metric=metric_name, verbose=False)
    metric_score_calibrated = predictor.evaluate(
        test_data, decision_threshold=calibrated_decision_threshold, silent=True
    )[metric_name]
```

### Usage Patterns

```python
y_pred = predictor.predict(test_data)  # Uses predictor.decision_threshold
y_pred_08 = predictor.predict(test_data, decision_threshold=0.8)  # Override threshold
y_pred_proba = predictor.predict_proba(test_data)
y_pred = predictor.predict_from_proba(y_pred_proba)  # Same as .predict()
y_pred_08 = predictor.predict_from_proba(y_pred_proba, decision_threshold=0.8)
```

**⚠️ Important:**
- Calibrating for one metric often **harms other metrics** — a tradeoff the user should consider
- Can auto-calibrate during fit: `predictor.fit(..., calibrate_decision_threshold=True)`
- Default `calibrate_decision_threshold="auto"` automatically applies calibration when beneficial — **recommended to keep as default**

## Prediction Options (Inference)

### Loading a Saved Predictor

```python
predictor = TabularPredictor.load(save_path)  # Also available via predictor.path
predictor.features()  # Required feature columns for predictions
```

Copy the `save_path` folder to deploy trained models on another machine.

### Making Predictions

```python
datapoint = test_data_nolabel.iloc[[0]]  # Must use [[0]] (DataFrame), not [0] (Series)
predictor.predict(datapoint)
predictor.predict_proba(datapoint)  # Class probabilities
```

### Model Selection & Leaderboard

```python
predictor.model_best  # Default model used for predictions (usually an ensemble)

# Evaluate all models on test data
predictor.leaderboard(test_data)

# Without test data (uses validation data from fit), with extra model info
predictor.leaderboard(extra_info=True)

# Additional metrics
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'])
```

**⚠️ Metrics are always in `higher_is_better` form** — `log_loss` and `root_mean_squared_error` will be **negated** (shown as negative values).

**⚠️ `log_loss` via `extra_metrics` can be `-inf`** when models (e.g., KNN) assign `0` probability to the correct class. Use `log_loss` only as `eval_metric`, not as a secondary metric.

### Predict with a Specific Model

```python
model_to_use = predictor.model_names()[0]
model_pred = predictor.predict(datapoint, model=model_to_use)
```

A "model" in AutoGluon can be a single model, a bagged ensemble, a weighted ensemble, or a stacker model.

### Evaluation & Model Info

```python
y_pred_proba = predictor.predict_proba(test_data_nolabel)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)
# Shorthand (if label column is in test_data):
perf = predictor.evaluate(test_data)

# Access model/predictor metadata
predictor_information = predictor.info()
```

## Feature Importance

```python
predictor.feature_importance(test_data)
```

Computed via **permutation-shuffling** — measures the drop in performance when a column's values are randomly shuffled. Features with **non-positive importance** may be harmful; consider removing them and re-fitting. These are global importance scores; for local (per-prediction) explanations, use [SHAP values](https://github.com/autogluon/autogluon/tree/master/examples/tabular/interpret).

## Accelerating Inference

### Optimization Priority

**With bagging** (`num_bag_folds>0`, `num_stack_levels>0`, or `best_quality`):
1. `refit_full` (8x–160x speedup, slight quality loss)
2. `persist` (up to 10x for online-inference)
3. `infer_limit` (configurable, up to 50x, quality tradeoff)

**Without bagging:**
1. `persist`
2. `infer_limit`

| Optimization | Speedup | Key Notes |
|:--|:--|:--|
| `refit_full` | 8x–160x | Only with bagging enabled |
| `persist` | Up to 10x | Online-inference only; requires sufficient memory |
| `infer_limit` | Up to 50x | Always use `refit_full` first if bagging enabled |
| `distill` | Comparable to refit_full + extreme infer_limit | Not compatible with refit_full/infer_limit |
| Feature pruning | Up to 1.5x | Use `predictor.feature_importance()` to identify removable features |

### Keeping Models in Memory

```python
predictor.persist()  # Load all required models into memory

for i in range(num_test):
    datapoint = test_data_nolabel.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)

predictor.unpersist()  # Free memory; future predict() loads from disk
```

Use `models='all'` to persist every trained model, or specify particular models via the `models` argument.

### Inference Speed as a Fit Constraint

Two parameters:
- **`infer_limit`**: seconds per row (e.g., `0.05` = 20 rows/sec)
- **`infer_limit_batch_size`**: rows per batch — `10000` for batch-inference (easier to satisfy), `1` for online-inference (`infer_limit<0.02` is difficult to satisfy with batch size 1)

```python
predictor_infer_limit = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data=train_data,
    time_limit=30,
    infer_limit=0.00005,  # 0.05ms per row = 20000 rows/sec
    infer_limit_batch_size=10000,
)

# IMPORTANT: If bagging was enabled, call refit_full (infer_limit assumes this)
# predictor_infer_limit.refit_full()

# Persist models to match inference speed calculated during fit
predictor_infer_limit.persist()
```

**⚠️ With bagging + `infer_limit`:** always call `refit_full()` after fit — `infer_limit` assumes this will happen. Models must be persisted to match the inference speed calculated during fit.

### Verifying Inference Speed

```python
test_data_batch = test_data.sample(infer_limit_batch_size, replace=True, ignore_index=True)
import time
time_start = time.time()
predictor_infer_limit.predict(test_data_batch)
time_end = time.time()
infer_time_per_row = (time_end - time_start) / len(test_data_batch)
print(f'Model uses {round((infer_time_per_row / infer_limit) * 100, 1)}% of infer_limit time per row.')
```

### Alternative Ensembles (Accuracy-Speed Tradeoffs)

```python
additional_ensembles = predictor.fit_weighted_ensemble(expand_pareto_frontier=True)
predictor.leaderboard(only_pareto_frontier=True)  # Most accurate model per latency tier

predictions = predictor.predict(test_data, model=additional_ensembles[0])
predictor.delete_models(models_to_delete=additional_ensembles)  # Cleanup
```

### Collapsing Bagged Ensembles via `refit_full`

Collapses ~10 bagged model copies into a single model trained on full data — greatly reduces memory/latency but may reduce accuracy. No `score_val` available for refit models since all data was used for training.

```python
refit_model_map = predictor.refit_full()  # Use model= argument for specific models
predictor.leaderboard(test_data)
```

Works with non-bagged models too (no latency gain, but potential accuracy improvement).

### Model Distillation

Train a single "student" model to mimic the full ensemble "teacher" — retains some ensemble accuracy with single-model speed.

```python
student_models = predictor.distill(time_limit=30)  # Use much longer in practice
preds_student = predictor.predict(test_data_nolabel, model=student_models[0])
```

**⚠️** Not compatible with `refit_full` and `infer_limit`.

### Faster Presets & Hyperparameters

```python
# Lightweight presets
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, presets=['good_quality', 'optimize_for_deployment'], time_limit=30)

# Lightweight hyperparameters: 'light', 'very_light', or 'toy' (progressively smaller)
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, hyperparameters='very_light', time_limit=30)

# Exclude slow model types
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, excluded_model_types=['KNN', 'NN_TORCH'], time_limit=30)
```


...(truncated)