# Condensed: Force features to be passed to models without preprocessing / dropping

Summary: This tutorial demonstrates how to prevent AutoGluon from dropping features during preprocessing, covering two levels: model-specific and global. Key techniques include overriding `_get_default_auxiliary_params` with `drop_unique=False` in custom `AbstractModel` subclasses, creating custom `BulkFeatureGenerator` subclasses that bifurcate preprocessing using `IdentityFeatureGenerator` for tagged features, and using `FeatureMetadata.add_special_types` to tag features with `user_override`. It helps with tasks involving custom feature generators, preserving constant-value features needed by custom models, and integrating these components into `TabularPredictor.fit()`. Important: custom classes must be in separate importable files for serialization.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon: Force Features Through Preprocessing

## Setup
```python
!pip install autogluon.tabular[all]
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)
```

## Preventing Feature Dropping

**Use case:** Features with single unique values (e.g., a language identifier when training data has one language) get dropped by default but may be needed by custom models.

### 1. Prevent Dropping in Model-Specific Preprocessing

Override `_get_default_auxiliary_params` with `drop_unique=False`:

```python
from autogluon.core.models import AbstractModel

class DummyModelKeepUnique(AbstractModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(dict(
            drop_unique=False,  # Default is True
        ))
        return default_auxiliary_params
```

### 2. Prevent Dropping in Global Preprocessing

> **⚠️ WARNING:** This class must be in a **separate Python file** from the main process and imported, or it won't be serializable.

Create a custom feature generator that bifurcates preprocessing based on `user_override` tag:

```python
from autogluon.features import BulkFeatureGenerator, AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator

class CustomFeatureGeneratorWithUserOverride(BulkFeatureGenerator):
    def __init__(self, automl_generator_kwargs: dict = None, **kwargs):
        generators = self._get_default_generators(automl_generator_kwargs=automl_generator_kwargs)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, automl_generator_kwargs: dict = None):
        if automl_generator_kwargs is None:
            automl_generator_kwargs = dict()
        generators = [[
            AutoMLPipelineFeatureGenerator(banned_feature_special_types=['user_override'], **automl_generator_kwargs),
            IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=['user_override'])),
        ]]
        return generators
```

Tag features with `user_override` via `FeatureMetadata`:

```python
train_data['dummy_feature'] = 'dummy value'
test_data['dummy_feature'] = 'dummy value'

from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({
    'age': ['user_override'],
    'native-country': ['user_override'],
    'dummy_feature': ['user_override'],
})
```

### 3. Putting It Together (Standalone)

```python
X = train_data.drop(columns=[label])
y = train_data[label]

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_preprocessed = label_cleaner.transform(y)

my_custom_feature_generator = CustomFeatureGeneratorWithUserOverride(feature_metadata_in=feature_metadata)
X_preprocessed = my_custom_feature_generator.fit_transform(X)
```

### 4. Via TabularPredictor

> **⚠️** Custom model and feature generator classes **must exist in separate importable files** for serialization.

```python
from autogluon.tabular import TabularPredictor

feature_generator = CustomFeatureGeneratorWithUserOverride()
predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,
    feature_generator=feature_generator,
    hyperparameters={
        'GBM': {},
        DummyModelKeepUnique: {},  # Won't drop dummy_feature
        # Alternative: DummyModel: {'ag_args_fit': {'drop_unique': False}},
    }
)
```

**Key takeaway:** `IdentityFeatureGenerator` is a no-op passthrough—replace it with any custom feature generator for more complex override preprocessing logic.