# Condensed: Snapshot a Predictor with .clone()

Summary: AutoGluon TabularPredictor deployment guide covering techniques for optimizing and deploying trained tabular ML models. Helps with: training/loading predictors, making predictions, snapshotting via `.clone()` for safe experimentation with rollback, creating minimal deployment artifacts via `.clone_for_deployment()` to reduce disk usage, persisting models in memory with `.persist()`, and compiling models to ONNX for faster inference via `.compile()`. Key details include disk usage comparison between original and optimized predictors, ONNX compilation limitations (RandomForest/TabularNeuralNetwork only), and best practices like version matching between training and inference environments and always compiling on cloned predictors.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon TabularPredictor Deployment Guide

## Setup & Training

```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
label = 'class'
train_data = train_data.sample(n=500, random_state=0)

save_path = 'agModels-predictClass-deployment'
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
```

Load and predict on test data:
```python
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = TabularPredictor.load(save_path)
y_pred = predictor.predict(test_data)
predictor.leaderboard(test_data)
```

## Snapshotting with `.clone()`

Creates an exact replica of the predictor at a new path, enabling safe experimentation with rollback capability.

```python
save_path_clone = save_path + '-clone'
path_clone = predictor.clone(path=save_path_clone)
predictor_clone = TabularPredictor.load(path=path_clone)
# Or: predictor_clone = predictor.clone(path=save_path_clone, return_clone=True)
```

> **Warning:** Cloning doubles disk usage as it replicates all artifacts.

Use case — safely call destructive operations like `refit_full()`, `delete_models()`, or `fit_extra()` on the clone while preserving the original.

## Deployment-Optimized Clone via `.clone_for_deployment()`

Creates a minimal clone containing only artifacts needed for prediction. **Cannot train additional models.**

```python
save_path_clone_opt = save_path + '-clone-opt'
path_clone_opt = predictor.clone_for_deployment(path=save_path_clone_opt)
predictor_clone_opt = TabularPredictor.load(path=path_clone_opt)

# Persist model in memory to avoid reloading on each predict call
predictor_clone_opt.persist()

y_pred_clone_opt = predictor_clone_opt.predict(test_data)
```

Check disk savings:
```python
size_original = predictor.disk_usage()
size_opt = predictor_clone_opt.disk_usage()
print(f'Reduction: {round((1 - (size_opt/size_original)) * 100, 1)}%')
```

Use `predictor.disk_usage_per_file()` to inspect file-level differences.

## Compile Models for Faster Inference

**Experimental feature** — converts sklearn models to ONNX equivalents. Currently supports **RandomForest** and **TabularNeuralNetwork** only.

```bash
pip install autogluon.tabular[skl2onnx]
# or for new installs: pip install autogluon.tabular[all,skl2onnx]
```

> **Important:** Always compile on a **cloned** predictor — compiled models cannot be used for further fitting.

```python
predictor_clone_opt.compile()
y_pred_compiled = predictor_clone_opt.predict(test_data)
```

> **Note:** Compiled predictions may differ slightly from originals due to ONNX conversion but should be very close.

## Deployment Best Practices

- Upload the optimized predictor to centralized storage (e.g., S3), download to target machine, and load
- **Ensure the same Python version and AutoGluon version** used during training are present at inference time to avoid instability
- Use `.persist()` to keep models in memory for repeated predictions
- Use `.clone_for_deployment()` over `.clone()` for production to minimize artifact size