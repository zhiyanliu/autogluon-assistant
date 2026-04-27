# Condensed: We use uv for faster installation

Summary: This tutorial covers AutoGluon's time series forecasting pipeline using `TimeSeriesDataFrame` and `TimeSeriesPredictor`. It demonstrates loading panel data in long format with `from_data_frame()` (specifying `id_column`, `timestamp_column`), configuring predictors with `prediction_length`, `eval_metric` (MASE), and `target`, training with quality presets (`fast_training` to `best_quality`) and `time_limit`, generating probabilistic multi-step forecasts with quantiles, visualizing predictions via `predictor.plot()`, and evaluating models with `predictor.leaderboard()`. Useful for implementing automated multi-series forecasting, model selection, and probabilistic prediction tasks with minimal code.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Time Series Forecasting - Condensed Tutorial

## Setup
```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system  # fix Colab incompatibilities
```

## Core Classes
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```
- **`TimeSeriesDataFrame`** — stores multiple time series
- **`TimeSeriesPredictor`** — fits, tunes, selects models, and generates forecasts

## Data Format

AutoGluon requires **long format** with three columns: unique ID, timestamp, and target value. Column names are flexible but must be specified.

```python
df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

**Key concept:** Each time series is an *item* (e.g., product, stock). This is **panel** forecasting, not multivariate — each series is forecast independently.

`TimeSeriesDataFrame` inherits from `pandas.DataFrame`, so all pandas methods are available.

## Training

`prediction_length` = number of future steps to forecast. Set based on task frequency (e.g., 48 for 48 hours ahead with hourly data).

```python
predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)
```

**Presets** (increasing quality/time): `"fast_training"`, `"medium_quality"`, `"high_quality"`, `"best_quality"`

`medium_quality` includes: `Naive`, `SeasonalNaive`, `ETS`, `Theta`, `RecursiveTabular`, `DirectTabular` (LightGBM), `TemporalFusionTransformer`, and a weighted ensemble.

**Validation:** By default, the last `prediction_length` timesteps of each series are held out for internal validation and model ranking.

## Prediction

```python
predictions = predictor.predict(train_data)
```

Produces **probabilistic forecasts**: mean predictions plus quantiles (e.g., `"0.1"` quantile = 10% chance target falls below that value). Forecasts the next `prediction_length` steps from each series' end.

## Visualization

```python
test_data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv")
predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)
```

## Evaluation

```python
predictor.leaderboard(test_data)
```

Test data must include both history and the forecast horizon (last `prediction_length` values). **Leaderboard scores are sign-flipped** (higher = better), so MASE appears as negative values.