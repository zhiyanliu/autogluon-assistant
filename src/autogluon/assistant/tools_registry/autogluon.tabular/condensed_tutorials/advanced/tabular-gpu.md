# Condensed: Multi-modal

Summary: This tutorial covers GPU and CPU resource allocation for AutoGluon's TabularPredictor, enabling tasks like configuring GPU acceleration at the predictor level (`num_gpus` in `fit()`), assigning per-model GPU/CPU resources via `ag_args_fit` in `hyperparameters`, and setting up multimodal configs using `get_hyperparameter_config('multimodal')`. It details a three-tier resource hierarchy (predictor → bagged model → base model) using `ag_args_ensemble` and `ag_args_fit`, with constraints that child resources must not exceed parent levels. It also covers enabling LightGBM GPU support and parallel HPO trial execution with `num_bag_folds` and `hyperparameter_tune_kwargs`.

*This is a condensed version that preserves essential implementation details and context.*

# GPU & Resource Allocation for TabularPredictor

## Basic GPU Usage

```python
# Grant 1 GPU to entire predictor
predictor = TabularPredictor(label=label).fit(train_data, num_gpus=1)
```

**Per-model GPU control** via `hyperparameters`:

```python
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # CPU
        {'ag_args_fit': {'num_gpus': 1}}   # GPU (must be <= total num_gpus)
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, num_gpus=1, hyperparameters=hyperparameters)
```

## Multi-modal

For multimodal (tabular + text + image), retrieve config with:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

## LightGBM GPU Support

Default LightGBM doesn't support GPU. If GPU fallback warning appears, reinstall from source:

```bash
pip uninstall -y lightgbm
# Follow official GPU build guide: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
# Complete the "Install Python Interface" section as well
```

## Advanced Resource Allocation

Three levels of resource control (each must be ≤ its parent level):

| Level | Parameter | Scope |
|-------|-----------|-------|
| **Predictor** | `num_cpus`, `num_gpus` in `fit()` | Total resources |
| **Bagged model** | `ag_args_ensemble={'ag_args_fit': {RESOURCES}}` | Per bagged model (ignored if bagging disabled) |
| **Base model** | `ag_args_fit={RESOURCES}` | Per individual fold/model |

```python
predictor.fit(
    num_cpus=32, num_gpus=4,
    hyperparameters={'NN_TORCH': {}},
    num_bag_folds=2,
    ag_args_ensemble={'ag_args_fit': {'num_cpus': 10, 'num_gpus': 2}},
    ag_args_fit={'num_cpus': 4, 'num_gpus': 0.5},
    hyperparameter_tune_kwargs={'searcher': 'random', 'scheduler': 'local', 'num_trials': 2}
)
```

**Result**: 2 HPO trials run in parallel (10 CPUs + 2 GPUs each), each training 2 folds in parallel (4 CPUs + 0.5 GPUs each) → **4 models training simultaneously**, using 16 CPUs and 2 GPUs total.