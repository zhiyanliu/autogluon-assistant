# Condensed: Ensuring Metric is Serializable

Summary: This tutorial demonstrates how to create and use custom evaluation metrics in AutoGluon using `make_scorer` from `autogluon.core.metrics`. It covers defining scorers for classification (`needs_class`), regression (`needs_pred`), threshold-based (`needs_threshold`), and probability-based (`needs_proba`) metrics, with key parameters including `score_func`, `optimum`, and `greater_is_better`. It explains score vs. error computation, internal score flipping for lower-is-better metrics, and the critical requirement that custom metrics be pickleable (defined in separate files). It shows how to pass custom metrics as `eval_metric` or `extra_metrics` in `TabularPredictor` for training and evaluation.

*This is a condensed version that preserves essential implementation details and context.*

# Custom Metrics in AutoGluon - Condensed Tutorial

## Setup
```python
!pip install autogluon.tabular[all]
```

## Serializability Requirement

**Critical**: Custom metrics must be defined in a separate Python file and imported (must be pickleable). Otherwise, parallel training with Ray will crash with `_pickle.PicklingError: Can't pickle`.

```python
# Define in my_metrics.py, then: from my_metrics import ag_accuracy_scorer
```

## Creating Custom Metrics with `make_scorer`

### Custom Accuracy (Classification)

```python
from autogluon.core.metrics import make_scorer
import sklearn.metrics

ag_accuracy_scorer = make_scorer(
    name='accuracy',
    score_func=sklearn.metrics.accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True
)
```

**Key Parameters:**
- `name`: Display name during training
- `score_func`: Function taking `(y_true, y_pred)` returning float
- `optimum`: Best possible value from the original metric
- `greater_is_better`: **Critical** - if wrong, AutoGluon optimizes for worst model
- `needs_*` options (only one can be True):
  - `needs_pred`: Regression predictions (default if none specified)
  - `needs_proba`: Probability estimates (e.g., log_loss, roc_auc_ovo)
  - `needs_class`: Class predictions (e.g., accuracy, f1, precision, recall)
  - `needs_threshold`: Decision certainty, binary only (e.g., roc_auc, average_precision)
  - `needs_quantile`: Quantile predictions

**Score vs Error:**
```python
ag_accuracy_scorer(y_true, y_pred)           # score (higher is better)
ag_accuracy_scorer.score(y_true, y_pred)     # alias
ag_accuracy_scorer.error(y_true, y_pred)     # error = sign*optimum - score
```

### Custom MSE (Regression, lower-is-better)

```python
ag_mean_squared_error_scorer = make_scorer(
    name='mean_squared_error',
    score_func=sklearn.metrics.mean_squared_error,
    optimum=0,
    greater_is_better=False
)
```

**Important**: Scorers always report in `greater_is_better=True` form internally. For `greater_is_better=False` metrics, scores are flipped (negative values).

**Custom function example:**
```python
def mse_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

ag_mse_custom = make_scorer(name='mean_squared_error', score_func=mse_func,
                            optimum=0, greater_is_better=False)
```

### Custom ROC AUC (Threshold metric)

```python
ag_roc_auc_scorer = make_scorer(
    name='roc_auc',
    score_func=sklearn.metrics.roc_auc_score,
    optimum=1,
    greater_is_better=True,
    needs_threshold=True
)
```

### Advanced Note on `optimum`
If metric is `greater_is_better=False` with optimal value `-2`: specify `optimum=-2`. Score becomes `sign * raw_value`, error = `sign * optimum - score`.

## Using Custom Metrics in TabularPredictor

```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'

# As extra evaluation metrics
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters='toy')
predictor.leaderboard(test_data, extra_metrics=[ag_roc_auc_scorer, ag_accuracy_scorer])

# As primary eval_metric (used for model selection/optimization)
predictor_custom = TabularPredictor(label=label, eval_metric=ag_roc_auc_scorer).fit(
    train_data, hyperparameters='toy'
)
predictor_custom.leaderboard(test_data)
```