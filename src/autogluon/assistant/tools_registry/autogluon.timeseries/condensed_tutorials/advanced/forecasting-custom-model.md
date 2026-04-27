# Condensed: We use uv for faster installation

Summary: This tutorial demonstrates how to wrap a custom time series model (NeuralForecast NHITS) as an AutoGluon-compatible model by subclassing `AbstractTimeSeriesModel`. It covers implementing required methods (`_fit`, `_predict`, `preprocess`), handling covariates (known/past/static, real-valued only), NaN imputation, data format conversion between AutoGluon's `TimeSeriesDataFrame` and NeuralForecast, quantile prediction output formatting, lazy imports, time limit enforcement via PyTorch-Lightning's `max_time`, and default hyperparameter configuration with GPU support. It also shows standalone debugging, integration with `TimeSeriesPredictor` for multi-model comparison via leaderboard/feature importance, and training multiple hyperparameter variants of custom models.

*This is a condensed version that preserves essential implementation details and context.*

# NHITS Model Wrapper for AutoGluon

## Model Class Setup

```python
class NHITSModel(AbstractTimeSeriesModel):
    # Enable covariate/static feature support
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True
```

## Preprocessing

NeuralForecast **cannot handle NaNs** — impute before passing data:

```python
def preprocess(self, data, known_covariates=None, is_train=False, **kwargs):
    data = data.fill_missing_values()  # forward-fill + backward-fill
    data = data.fill_missing_values(method="constant", value=0.0)  # all-NaN series
    return data, known_covariates
```

## Default Hyperparameters

```python
def _get_default_hyperparameters(self) -> dict:
    from neuralforecast.losses.pytorch import MQLoss
    default_hyperparameters = dict(
        loss=MQLoss(quantiles=self.quantile_levels),
        input_size=2 * self.prediction_length,
        scaler_type="standard",
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        start_padding_enabled=True,  # handles single-observation series
        # Only real-valued covariates — NeuralForecast doesn't support categorical
        futr_exog_list=self.covariate_metadata.known_covariates_real,
        hist_exog_list=self.covariate_metadata.past_covariates_real,
        stat_exog_list=self.covariate_metadata.static_features_real,
    )
    if torch.cuda.is_available():
        default_hyperparameters["devices"] = 1
    return default_hyperparameters
```

## Fit

- **Lazy imports** inside `_fit` to reduce import time and isolate dependency issues
- **Time limit** enforced via PyTorch-Lightning's `max_time`
- `get_hyperparameters()` merges defaults with user-provided overrides

```python
def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    hyperparameter_overrides = {"max_time": {"seconds": time_limit}} if time_limit else {}
    model_params = self.get_hyperparameters() | hyperparameter_overrides

    model = NHITS(h=self.prediction_length, **model_params)
    self.nf = NeuralForecast(models=[model], freq=self.freq)

    train_df, static_df = self._to_neuralforecast_format(train_data)
    self.nf.fit(train_df, static_df=static_df, id_col="item_id",
                time_col="timestamp", target_col=self.target)
```

## Data Conversion

Drop categorical covariates (unsupported by NeuralForecast):

```python
def _to_neuralforecast_format(self, data):
    df = data.to_data_frame().reset_index()
    df = df.drop(columns=self.covariate_metadata.covariates_cat)
    static_df = data.static_features
    if len(self.covariate_metadata.static_features_real) > 0:
        static_df = static_df.reset_index().drop(columns=self.covariate_metadata.static_features_cat)
    return df, static_df
```

## Predict

Output must be a `TimeSeriesDataFrame` with columns `["mean"] + [str(q) for q in quantile_levels]`:

```python
def _predict(self, data, known_covariates=None, **kwargs):
    from neuralforecast.losses.pytorch import quantiles_to_outputs

    df, static_df = self._to_neuralforecast_format(data)
    futr_df = self._to_neuralforecast_format(known_covariates)[0] \
        if self.covariate_metadata.known_covariates_real else None

    predictions = self.nf.predict(df, static_df=static_df, futr_df=futr_df)

    model_name = str(self.nf.models[0])
    rename_columns = {
        f"{model_name}{suffix}": str(q)
        for q, suffix in zip(*quantiles_to_outputs(self.quantile_levels))
    }
    predictions = predictions.rename(columns=rename_columns)
    predictions["mean"] = predictions["0.5"]
    return TimeSeriesDataFrame(predictions)
```

**Key warnings:** NeuralForecast requires real-valued covariates only — use one-hot encoding or models with native categorical support for categorical features. The `preprocess` method runs automatically on all data passed to `_fit` and `_predict`.

# Using the Custom NHITS Model

## Setup

```bash
pip install autogluon.timeseries neuralforecast==2.0
```

## Custom Model Requirements

Subclass `AbstractTimeSeriesModel` and implement:
- **`_fit`** and **`_predict`** (required)
- **`preprocess`** (optional, e.g., for missing value handling)

## Data Loading & Preprocessing

```python
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

raw_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    static_features_path="https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/static.csv",
)

prediction_length = 7
target = "unit_sales"
known_covariates_names = ["promotion_email", "promotion_homepage"]

# TimeSeriesFeatureGenerator normalizes dtypes and imputes covariate missing values
feature_generator = TimeSeriesFeatureGenerator(target=target, known_covariates_names=known_covariates_names)
data = feature_generator.fit_transform(raw_data)
```

## Standalone Mode (for debugging)

```python
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    covariate_metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)
model.fit(train_data=data, time_limit=20)

past_data, known_covariates = data.get_model_inputs_for_scoring(
    prediction_length=prediction_length,
    known_covariates_names=known_covariates_names,
)
predictions = model.predict(past_data, known_covariates)
model.score(data)
```

## Inside TimeSeriesPredictor

The predictor **automatically handles** model configuration (`freq`, `prediction_length`), data preprocessing, and time limits.

```python
from autogluon.timeseries import TimeSeriesPredictor

train_data, test_data = raw_data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)
predictor.fit(
    train_data,
    hyperparameters={
        "Naive": {},
        "Chronos": {"model_path": "bolt_small"},
        "ETS": {},
        NHITSModel: {},  # custom model alongside built-in models
    },
    time_limit=120,
)
predictor.leaderboard(test_data)
predictor.feature_importance(test_data, model="NHITS")
```

**Note:** Categorical features show zero importance since this wrapper ignores them.

## Multiple Hyperparameter Configurations

Pass a list of dicts to train multiple variants:

```python
predictor.fit(
    train_data,
    hyperparameters={
        NHITSModel: [
            {},                          # defaults
            {"input_size": 20},          # custom input_size
            {"scaler_type": "robust"},   # custom scaler_type
        ]
    },
    time_limit=60,
)
```

## Key Takeaways

- Standalone mode is useful for **debugging** before integrating with the predictor
- The predictor enables easy **comparison** with built-in models via `leaderboard()` and access to `feature_importance()`
- Consider [submitting a PR](https://github.com/autogluon/autogluon/pulls) for community-useful custom models