# Condensed: Automatic Feature Engineering ##

Summary: This tutorial covers AutoGluon's automatic feature engineering pipeline, detailing how each column type is transformed: categorical→integer encoding, datetime→extracted temporal features (year/month/day/dayofweek), and text→n-gram n-hot encoding plus special features (word/char counts). It demonstrates using `AutoMLPipelineFeatureGenerator` and building custom pipelines with `PipelineFeatureGenerator`, `CategoryFeatureGenerator`, and `IdentityFeatureGenerator`. Key practical knowledge includes handling missing values (NaNs retained except datetime→mean-filled), explicitly casting integers as categorical via `.astype("category")`, filtering rare categories with `maximum_num_cat`, and integrating custom feature generators into `TabularPredictor.fit()`.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Feature Engineering

## Automatic Feature Engineering by Column Type

**Numerical:** No automated feature engineering applied.

**Categorical:** Mapped to monotonically increasing integers for downstream model compatibility.

**Datetime:** Converted to:
- Numerical Pandas datetime (bounded by `pandas.Timestamp.min`/`max`)
- Extracted columns: `[year, month, day, dayofweek]` (configurable via `DatetimeFeatureGenerator`)
- Missing/invalid/out-of-range values → replaced with mean of valid rows

**Text:** If [MultiModal](tabular-multimodal.ipynb) enabled, uses Transformer NLP models. Otherwise:
- **N-gram features** (`TextNgramFeatureGenerator`): All text columns concatenated, word-level n-grams extracted as n-hot encoded columns
- **Special features** (`TextSpecialFeatureGenerator`): Word count, character count, uppercase proportion, etc.

**Additional:** Columns with only 1 unique value or duplicate columns are dropped.

## Feature Engineering Example

```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

Results: float/int columns unchanged; datetime → raw nanoseconds + year/month/day/dayofweek; categorical → integers; text → summary features + n-hot word matrix.

**⚠️ Important:** Integer columns with few unique values are **not** auto-detected as categorical. Explicitly cast them:

```python
dfx["B"] = dfx["B"].astype("category")
```

## Missing Value Handling

Float, integer, categorical, and text fields **retain NaNs**. Datetime columns are **filled with the mean** of non-NaN values.

## Custom Feature Engineering

Use `PipelineFeatureGenerator` with non-default parameters:

```python
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
from autogluon.common.features.types import R_INT, R_FLOAT

mypipeline = PipelineFeatureGenerator(
    generators=[[
        CategoryFeatureGenerator(maximum_num_cat=10),  # Replace rare categories with NaN
        IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    ]]
)
mypipeline.fit_transform(X=dfx)
```

Pass custom pipeline to predictor:
```python
predictor = TabularPredictor(label='label')
predictor.fit(df, hyperparameters={'GBM': {}}, feature_generator=auto_ml_pipeline_feature_generator)
```

See `examples/tabular/example_custom_feature_generator.py` for more details.