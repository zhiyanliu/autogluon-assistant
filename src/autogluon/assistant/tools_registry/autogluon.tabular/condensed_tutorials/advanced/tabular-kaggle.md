# Condensed: ```

Summary: This tutorial demonstrates using AutoGluon's `TabularPredictor` for a Kaggle fraud detection competition, covering the full pipeline: merging multi-file CSV datasets via pandas joins, training with `presets='best_quality'` and `time_limit`, and generating predictions. Key techniques include using `predict_proba` with `as_multiclass=False` for AUC-based evaluation, checking `positive_class`/`class_labels` for correct probability alignment, and formatting Kaggle submissions. It also covers best practices: specifying `eval_metric`, handling time-based validation splits, and advanced `fit()` parameters (`num_bag_folds`, `num_stack_levels`, `hyperparameters`). Useful for implementing AutoGluon tabular classification workflows and Kaggle submission pipelines.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon for Kaggle IEEE Fraud Detection

## Setup & Data Loading

```python
import pandas as pd
from autogluon.tabular import TabularPredictor

directory = '~/IEEEfraud/'
label = 'isFraud'
eval_metric = 'roc_auc'
save_path = directory + 'AutoGluonModels/'

train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

> **Warning:** The join strategy (left-join on `TransactionID`) is competition-specific. Always carefully consider the appropriate join for your dataset.

## Training

Use `presets='best_quality'` for maximum accuracy (increase `time_limit` beyond 3600s for better results):

```python
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data, presets='best_quality', time_limit=3600
)
```

## Prediction & Submission

**Critical:** Join test files identically to training files. Use `predict_proba` (not `predict`) for AUC-based competitions.

```python
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# Check which class probabilities correspond to
predictor.positive_class          # binary classification
predictor.class_labels            # multiclass (columns of predict_proba output)

# Get positive-class probabilities only
y_predproba = predictor.predict_proba(test_data, as_multiclass=False)

submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.to_csv(directory+'my_submission.csv', index=False)
```

Submit: `kaggle competitions submit -c ieee-fraud-detection -f sample_submission.csv -m "my first submission"`

## Tips for Best Performance

- **Always specify the competition's eval metric**; if unsure, omit it and let AutoGluon infer.
- **Time-based data:** If test data is from the future, reserve the most recent training examples as a validation set passed to `fit()`. Otherwise, AutoGluon handles train/val splitting automatically.
- **Recommended approach:** Focus on feature engineering and use `presets='best_quality'`. Advanced users can tune `num_bag_folds`, `num_stack_levels`, `num_bag_sets`, `hyperparameter_tune_kwargs`, `hyperparameters`, `refit_full`.