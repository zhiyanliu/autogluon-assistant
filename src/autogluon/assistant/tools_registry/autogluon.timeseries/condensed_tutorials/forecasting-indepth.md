# Condensed: We use uv for faster installation

Summary: This tutorial covers AutoGluon's `TimeSeriesPredictor` and `TimeSeriesDataFrame` for multi-series forecasting. It demonstrates adding static features, known covariates (e.g., weekend indicators, holidays), and past covariates, including a reusable `add_holiday_features` helper function. It explains handling irregular data and missing values via `convert_frequency` and `fill_missing_values`, train/test splitting with `train_test_split`, evaluation with `evaluate`, and internal validation with `num_val_windows` or custom `tuning_data`. It covers model categories (local, global, ensemble), quality presets (`fast_training` through `best_quality`), manual model configuration via `hyperparameters`, model exclusion, and hyperparameter tuning with `space` definitions and `hyperparameter_tune_kwargs`.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Time Series: Covariates, Static Features & Holidays

## Setup
```python
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```

## Loading Data with Static Features

Static features are per-item metadata (e.g., category/domain). Pass them as a DataFrame with an `item_id` column mapping to time series items:

```python
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,  # DataFrame with item_id + feature columns
)
# Or attach to existing TimeSeriesDataFrame:
train_data.static_features = static_features_df
```

## Covariates: Known vs Past

Add columns directly to the `TimeSeriesDataFrame`:

```python
train_data["log_target"] = np.log(train_data["target"])  # past covariate

WEEKEND_INDICES = [5, 6]
timestamps = train_data.index.get_level_values("timestamp")
train_data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)  # known covariate
```

Specify known covariates when creating the predictor — **remaining columns (except target and known covariates) are automatically treated as past covariates**:

```python
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"],
).fit(train_data)
```

## Prediction with Known Covariates

Generate future known covariates for the forecast horizon:

```python
known_covariates = predictor.make_future_data_frame(train_data)
known_covariates["weekend"] = known_covariates["timestamp"].dt.weekday.isin(WEEKEND_INDICES).astype(float)
predictions = predictor.predict(train_data, known_covariates=known_covariates)
```

**`known_covariates` requirements:**
- Must include all columns in `predictor.known_covariates_names`
- Must include all `item_id`s present in `train_data`
- Must cover `prediction_length` time steps beyond each series' end

Extra columns/rows/timestamps are automatically filtered.

## Holiday Features

Define holidays via the `holidays` package or a custom dict (`{datetime.date: name}`):

```python
import holidays
country_holidays = holidays.country_holidays(
    country="DE",
    years=range(timestamps.min().year, timestamps.max().year + 1),
)
```

Helper to add holiday columns:

```python
def add_holiday_features(ts_df, country_holidays, include_individual_holidays=True, include_holiday_indicator=True):
    ts_df = ts_df.copy()
    if not isinstance(ts_df, TimeSeriesDataFrame):
        ts_df = TimeSeriesDataFrame(ts_df)
    timestamps = ts_df.index.get_level_values("timestamp")
    country_holidays_df = pd.get_dummies(pd.Series(country_holidays)).astype(float)
    holidays_df = country_holidays_df.reindex(timestamps.date).fillna(0)
    if include_individual_holidays:
        ts_df[holidays_df.columns] = holidays_df.values
    if include_holiday_indicator:
        ts_df["Holiday"] = holidays_df.max(axis=1).values
    return ts_df
```

**Training** — register holiday columns as known covariates:

```python
train_data_with_holidays = add_holiday_features(train_data, country_holidays)
holiday_columns = train_data_with_holidays.columns.difference(train_data.columns)
predictor = TimeSeriesPredictor(..., known_covariates_names=holiday_columns).fit(train_data_with_holidays)
```

**Prediction** — provide future holiday values:

```python
known_covariates = predictor.make_future_data_frame(train_data)
known_covariates = add_holiday_features(known_covariates, country_holidays)
predictions = predictor.predict(train_data_with_holidays, known_covariates=known_covariates)
```

> **Note:** Not all models support static features and covariates — check the [Forecasting Model Zoo](forecasting-model-zoo.md) for compatibility.

# AutoGluon Time Series: Data Format, Missing Values & Evaluation

## Data Format Requirements

**Minimum time series length:**
- Default: at least some series must have length `>= max(prediction_length + 1, 5) + prediction_length`
- With advanced validation: `>= max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size`

Time series can have different lengths.

## Handling Irregular Data & Missing Values

For irregular time indices (e.g., missing weekends in financial data), specify frequency explicitly:

```python
predictor = TimeSeriesPredictor(..., freq="D").fit(df_irregular)
```

AutoGluon auto-converts irregular data and handles missing values. Alternatively, manually convert:

```python
df_regular = df_irregular.convert_frequency(freq="D")  # fills gaps with NaN
```

Fill strategies:
```python
df_filled = df_regular.fill_missing_values()  # default: forward + backward fill
df_filled = df_regular.fill_missing_values(method="constant", value=0.0)  # e.g., demand forecasting
```

Most AutoGluon models handle NaN natively, so manual filling is optional.

## Evaluating Forecast Accuracy

### Train/Test Split

```python
prediction_length = 48
data = TimeSeriesDataFrame.from_path("...")
train_data, test_data = data.train_test_split(prediction_length)
```

- `test_data` = full original data (history + forecast horizon)
- `train_data` = original data with last `prediction_length` steps removed per series

### Evaluation

```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE").fit(train_data)
predictor.evaluate(test_data)
```

**Key detail:** `evaluate` always scores on the last `prediction_length` time steps of each series in `test_data`. Earlier steps are used only to initialize models before forecasting.

### Internal Validation

AutoGluon automatically splits `train_data` into internal train/validation sets. The best-performing model on validation is used for prediction.

**Multiple validation windows** (reduces overfitting, increases training time proportionally):
```python
predictor.fit(train_data, num_val_windows=3)
```
Requires series length `>= (num_val_windows + 1) * prediction_length`.

**Custom validation set** (score computed on last `prediction_length` steps):
```python
predictor.fit(train_data=train_data, tuning_data=my_validation_dataset)
```

# AutoGluon Time Series: Models, Presets & Configuration

## Available Model Categories

**Local models** — simple statistical models, fit separately per time series (re-fit from scratch for new series):
- `ETS`, `AutoARIMA`, `Theta`, `SeasonalNaive`

**Global models** — single model learned across all time series:
- Neural networks (via GluonTS/PyTorch): `DeepAR`, `PatchTST`, `DLinear`, `TemporalFusionTransformer`
- Pre-trained zero-shot: Chronos
- Tabular: `RecursiveTabular`, `DirectTabular` (convert forecasting → regression, use LightGBM via `autogluon.tabular`)

**Ensemble** — `WeightedEnsemble` combines all model predictions (enabled by default; disable with `enable_ensemble=False`).

## Presets & Time Limit

```python
predictor = TimeSeriesPredictor(...)
predictor.fit(train_data, presets="medium_quality")
```

| Preset | Description | Relative Fit Time |
|:--|:--|:--|
| `fast_training` | Statistical + fast tree-based models | 0.5x |
| `medium_quality` | Above + TFT + Chronos-Bolt (small) | 1x |
| `high_quality` | More powerful DL, ML, statistical & pretrained models | 3x |
| `best_quality` | Same as `high_quality` + more CV windows (best for <50 series) | 6x |

Control training time directly:
```python
predictor.fit(train_data, time_limit=60 * 60)  # seconds; trains until all models fit if omitted
```

## Manual Model Configuration

Override presets via `hyperparameters` — pass `{}` for defaults, or a list for multiple configs:

```python
predictor.fit(
    ...,
    hyperparameters={
        "DeepAR": {},  # default hyperparameters
        "Theta": [
            {"decomposition_type": "additive"},
            {"seasonal_period": 1},
        ],
    }
)
```

Exclude specific models from presets:
```python
predictor.fit(..., presets="high_quality", excluded_model_types=["AutoETS", "AutoARIMA"])
```

## Hyperparameter Tuning

```python
from autogluon.common import space

predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {
            "hidden_size": space.Int(20, 100),
            "dropout_rate": space.Categorical(0.1, 0.3),
        },
    },
    hyperparameter_tune_kwargs="auto",  # 10 trials by default
    enable_ensemble=False,
)
```

Custom HPO configuration:
```python
predictor.fit(
    ...,
    hyperparameter_tune_kwargs={
        "num_trials": 20,        # configs per tuned model
        "searcher": "random",    # only supported option
        "scheduler": "local",    # only supported option
    },
)
```

Uses Ray Tune for GluonTS deep learning models, random search for others.

> **Warning:** HPO significantly increases training time but often provides only modest performance gains.