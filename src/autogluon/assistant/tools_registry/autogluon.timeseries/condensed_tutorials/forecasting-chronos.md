# Condensed: We use uv for faster installation

Summary: This tutorial covers using Chronos and Chronos-Bolt pretrained models for time series forecasting via AutoGluon's `TimeSeriesPredictor`. It demonstrates three key workflows: **zero-shot forecasting** using presets (e.g., `"bolt_small"`), **fine-tuning** Chronos on custom data with configurable learning rate/steps via `hyperparameters`, and **incorporating covariates** with univariate Chronos using `covariate_regressor` (e.g., CatBoost) and `target_scaler`. Key implementation details include `TimeSeriesDataFrame` data loading, `train_test_split`, model comparison via `leaderboard()`, and specifying `known_covariates_names`. Useful for building forecasting pipelines with pretrained foundation models, model evaluation, and handling exogenous variables.

*This is a condensed version that preserves essential implementation details and context.*

# Chronos in AutoGluon-TimeSeries: Condensed Tutorial

## Setup
```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system  # fix Colab incompatibilities
```

## Key Concepts

- Chronos models **don't fit** data — all computation happens at inference (`predict`), scaling linearly with number of time series (like ETS/ARIMA).
- **Chronos-Bolt⚡** (recommended): Up to 250x faster, more accurate. Presets: `"bolt_tiny"`, `"bolt_mini"`, `"bolt_small"`, `"bolt_base"`. Runs on **CPU or GPU**.
- **Original Chronos**: Presets: `"chronos_tiny"` through `"chronos_large"`. Sizes `small`+ **require GPU**.
- Can combine with other models via `"medium_quality"`, `"high_quality"`, `"best_quality"` presets.

## Zero-Shot Forecasting

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)

prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small",
)

predictions = predictor.predict(train_data)
predictor.plot(data=data, predictions=predictions, item_ids=data.item_ids[:2], max_history_length=200)
```

## Fine-Tuning

Compare zero-shot vs fine-tuned by passing both configs:

```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,
    enable_ensemble=False,
)
predictor.leaderboard(test_data)
```

Custom fine-tuning parameters:
```python
hyperparameters={"Chronos": {"fine_tune": True, "fine_tune_lr": 1e-4, "fine_tune_steps": 2000}}
```

> **Note:** AG-TS reports scores in "higher is better" format — error metrics like WQL are multiplied by -1.

## Incorporating Covariates

Chronos is univariate, but **covariate regressors** can be combined with it. The regressor predicts target from covariates/static features; its predictions are subtracted, and Chronos forecasts the residuals.

```python
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
)
prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length=prediction_length)

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=60,
)
predictor.leaderboard(test_data)
```

> **Best practice:** Always use `target_scaler` (e.g., `"standard"`) with covariate regressors to normalize time series scales.

> **Warning:** Covariates aren't always beneficial — always compare against zero-shot. `"high_quality"` and `"best_quality"` presets handle model selection automatically.

## FAQ

- **Accuracy**: Chronos-Bolt (base) often exceeds statistical baselines and is comparable to deep learning models (TFT, PatchTST).
- **Hardware**: For fine-tuning/inference with larger models, use GPU instances with ≥16GiB GPU memory and ≥32GiB RAM (e.g., AWS `g5.2xlarge`, `p3.2xlarge`). Bolt models work on CPU but slower.
- **Support**: [AutoGluon Discord](https://discord.gg/wjUmjqAc2N), [GitHub](https://github.com/autogluon/autogluon), [Chronos Discussions](https://github.com/amazon-science/chronos-forecasting/discussions).