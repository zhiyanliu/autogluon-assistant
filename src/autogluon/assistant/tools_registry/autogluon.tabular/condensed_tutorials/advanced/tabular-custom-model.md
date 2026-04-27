# Condensed: Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None

Summary: This tutorial demonstrates how to implement custom models in AutoGluon by extending `AbstractModel`, covering the four key methods: `_preprocess` (feature transformation with stateful encoders), `_fit` (model training with problem-type handling), `_set_default_params` (default hyperparameters), and `_get_default_auxiliary_params` (valid input dtypes). It covers standalone training workflows including manual label cleaning (`LabelCleaner`), feature generation (`AutoMLPipelineFeatureGenerator`), and prediction/scoring. Advanced usage includes wrapping custom models in `BaggedEnsembleModel` for k-fold bagging, training multiple hyperparameter configurations via `TabularPredictor`, HPO with `space.Int/Real/Categorical` search spaces, and integrating custom models alongside default AutoGluon models using `get_hyperparameter_config`.

*This is a condensed version that preserves essential implementation details and context.*

# Custom Model Implementation in AutoGluon

## Installation
```python
!pip install autogluon.tabular[all]
```

## Custom Model Class

Extend `AbstractModel` to create custom models. Key methods to implement:

```python
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model_cls = RandomForestRegressor if self.problem_type in ['regression', 'softclass'] else RandomForestClassifier
        X = self.preprocess(X, is_train=True)  # Must call preprocess with is_train=True
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {'n_estimators': 300, 'n_jobs': -1, 'random_state': 0}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            # Raw dtypes: ['int', 'float', 'category', 'object', 'datetime']
            # object = raw text/image paths (specialized models only); datetime usually pre-converted to int
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

**Key implementation notes:**
- **`_preprocess`**: Called during both fit and predict; use `is_train` flag to fit stateful transformers only once. `LabelEncoderFeatureGenerator` converts categoricals to numeric. Fill NaN for algorithms that can't handle missing values (LightGBM can handle NaN natively).
- **`_fit`**: Import dependencies **inside the method** for modularity. Call `self.preprocess(X, is_train=True)` early. Set `self.model` to the trained model. Valid `problem_type` values: `['binary', 'multiclass', 'regression', 'quantile', 'softclass']`.
- **`_set_default_params`**: User-specified params override defaults key-by-key.
- **`_get_default_auxiliary_params`**: Controls valid input dtypes and other model-agnostic settings. Also supports `valid_special_types`, `ignored_type_group_raw`, `ignored_type_group_special`.

## Data Loading & Standalone Training

```python
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)
```

### Label Cleaning (required before standalone training)

Convert string labels to numeric for binary classification:

```python
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

X, y = train_data.drop(columns=[label]), train_data[label]
X_test, y_test = test_data.drop(columns=[label]), test_data[label]

problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)
```

> Training outside `TabularPredictor` is useful for **debugging** custom models with minimal code. The process mirrors internal `TabularPredictor.fit()` behavior in simplified form.

# Custom Model Training & Advanced Usage

## Feature Cleaning

Use `AutoMLPipelineFeatureGenerator` to convert object dtypes to categorical and minimize memory:

```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)
```

**Warning:** `AutoMLPipelineFeatureGenerator` does **not** fill missing numeric values, rescale numerics, or one-hot encode categoricals. Add these operations in your `_preprocess` method if needed.

## Fit, Predict & Score

```python
custom_model = CustomRandomForestModel()
# custom_model = CustomRandomForestModel(hyperparameters={'max_depth': 10})  # override defaults
custom_model.fit(X=X_clean, y=y_clean)

# Predict (apply same transforms to test data)
X_test_clean = feature_generator.transform(X_test)
y_test_clean = label_cleaner.transform(y_test)

y_pred = custom_model.predict(X_test_clean)
y_pred_orig = label_cleaner.inverse_transform(y_pred)  # Convert back to original labels

score = custom_model.score(X_test_clean, y_test_clean)  # Default: accuracy for binary classification
```

## Bagged Custom Model (without TabularPredictor)

Quick quality improvement via bagging:

```python
from autogluon.core.models import BaggedEnsembleModel

bagged_custom_model = BaggedEnsembleModel(CustomRandomForestModel())
# Required if custom model class is defined in notebook (pickle serialization issue)
bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local'
bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)
```

**Note:** Put custom model in a **separate module** to enable parallel fold fitting. The bagged model averages predictions from all k-fold models.

## Training with TabularPredictor

No need for manual LabelCleaner, FeatureGenerator, or validation set:

```python
from autogluon.tabular import TabularPredictor

# Train 3 models with different hyperparameters
custom_hyperparameters = {CustomRandomForestModel: [{}, {'max_depth': 10}, {'max_features': 0.9, 'max_depth': 20}]}
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)

predictor.leaderboard(test_data)  # Includes auto-trained WeightedEnsemble
y_pred = predictor.predict(test_data)  # Uses best model automatically
# y_pred = predictor.predict(test_data, model='CustomRandomForestModel_3')  # Specific model
```

## Hyperparameter Tuning

Define search spaces using `autogluon.common.space`:

```python
from autogluon.common import space

custom_hyperparameters_hpo = {CustomRandomForestModel: {
    'max_depth': space.Int(lower=5, upper=30),
    'max_features': space.Real(lower=0.1, upper=1.0),
    'criterion': space.Categorical('gini', 'entropy'),
}}

predictor = TabularPredictor(label=label).fit(
    train_data,
    hyperparameters=custom_hyperparameters_hpo,
    hyperparameter_tune_kwargs='auto',
    time_limit=20,
)
```

HPO trial models appear with `'/Tx'` suffix in leaderboard. Retrieve best model hyperparameters:

```python
leaderboard_hpo = predictor.leaderboard()
best_model_name = leaderboard_hpo[leaderboard_hpo['stack_level'] == 1]['model'].iloc[0]
best_model_info = predictor.info()['model_info'][best_model_name]
print(best_model_info['hyperparameters'])
```

## Training Custom Model Alongside Default Models

Merge custom model into default hyperparameter config:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
```

# Training Custom Model Alongside Default Models

Merge tuned custom model hyperparameters with AutoGluon defaults:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

custom_hyperparameters = get_hyperparameter_config('default')
custom_hyperparameters[CustomRandomForestModel] = best_model_info['hyperparameters']

predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)

# For multi-layer stack ensemble with custom model:
# predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters, presets='best_quality')

predictor.leaderboard(test_data)
```

## Summary

- Custom models inherit from `AbstractModel` and implement `_preprocess`, `_fit`, `_set_default_params`, and `_get_default_auxiliary_params`
- Can be used standalone, bagged, with HPO, or alongside default models
- Consider [submitting a PR](https://github.com/autogluon/autogluon/pulls) for community-useful custom models
- For advanced custom models, see [Adding a custom model (Advanced)](tabular-custom-model-advanced.ipynb)