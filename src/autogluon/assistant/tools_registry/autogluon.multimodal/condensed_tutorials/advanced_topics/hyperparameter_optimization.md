# Condensed: The Regular Model Fitting

Summary: This tutorial demonstrates hyperparameter optimization (HPO) with AutoGluon's `MultiModalPredictor` for image classification tasks. It covers regular model fitting with default settings, then configuring HPO using Ray Tune search spaces (`tune.uniform`, `tune.choice`) for parameters like learning rate, optimizer type, epochs, and model backbone. Key HPO control options include searcher (`"bayes"`, `"random"`), scheduler (`"ASHA"`, `"FIFO"`), `num_trials`, and `num_to_keep`. It helps with coding tasks involving automated model selection, hyperparameter tuning for vision models (timm checkpoints), and evaluation using `MultiModalPredictor.fit()` with `hyperparameter_tune_kwargs`.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal HPO Tutorial (Condensed)

## Setup & Data

```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset
from ray import tune

download_dir = './ag_automm_tutorial_hpo'
train_data, test_data = shopee_dataset(download_dir)
train_data = train_data.sample(frac=0.5)  # 400 data points, columns: image (path), label
```

## Regular Model Fitting

```python
predictor_regular = MultiModalPredictor(label="label")
predictor_regular.fit(
    train_data=train_data,
    hyperparameters={"model.timm_image.checkpoint_name": "ghostnet_100"}
)
scores = predictor_regular.evaluate(test_data, metrics=["accuracy"])
```

## HPO Configuration

Uses [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) backend. Accepts Tune or AutoGluon search spaces.

**1. Hyperparameter Search Space:**

```python
hyperparameters = {
    "optim.lr": tune.uniform(0.00005, 0.005),
    "optim.optim_type": tune.choice(["adamw", "sgd"]),
    "optim.max_epochs": tune.choice(["10", "20"]),
    "model.timm_image.checkpoint_name": tune.choice(["swin_base_patch4_window7_224", "convnext_base_in22ft1k"])
}
```

**2. HPO Control (`hyperparameter_tune_kwargs`):**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `searcher` | `"random"`, `"bayes"` | Search strategy |
| `scheduler` | `"FIFO"`, `"ASHA"` | Job scheduling |
| `num_trials` | int | Number of HPO trials |
| `num_to_keep` | int (default: 3, min: 1) | Checkpoints kept per trial |

## Full HPO Example

```python
predictor_hpo = MultiModalPredictor(label="label")

hyperparameters = {
    "optim.lr": tune.uniform(0.00005, 0.001),
    "model.timm_image.checkpoint_name": tune.choice(["ghostnet_100", "mobilenetv3_large_100"])
}
hyperparameter_tune_kwargs = {
    "searcher": "bayes",
    "scheduler": "ASHA",
    "num_trials": 2,
    "num_to_keep": 3,
}

predictor_hpo.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
scores_hpo = predictor_hpo.evaluate(test_data, metrics=["accuracy"])
```

**Key takeaway:** HPO selects the hyperparameter combination with highest validation accuracy. Even a simple 2-trial run can outperform default settings by searching over learning rates and model architectures.