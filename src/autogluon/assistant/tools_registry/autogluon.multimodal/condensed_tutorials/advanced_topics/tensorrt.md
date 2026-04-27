# Condensed: Install required packages

Summary: This tutorial demonstrates how to optimize AutoGluon-MultiModal inference using TensorRT via `predictor.optimize_for_inference()`, which converts PyTorch models to ONNX Runtime-based modules. It covers installing tensorrt/onnx/onnxruntime dependencies, training a multimodal predictor (image, text, tabular) with configurable backbones (timm_image, hf_text, MLPs, fusion_mlp), and applying TensorRT optimization to a saved model. Key details include FP16 mixed precision behavior (atol=0.01), switching execution providers like `CUDAExecutionProvider` to disable mixed precision, and the critical warning that `optimize_for_inference()` makes the predictor inference-only—requiring a fresh `MultiModalPredictor.load` for retraining.

*This is a condensed version that preserves essential implementation details and context.*

# TensorRT Inference Optimization with AutoGluon-MultiModal

AutoGluon-MultiModal integrates with TensorRT via `predictor.optimize_for_inference()` to boost inference speed for deployment.

## Setup

Required optional dependencies:

```python
try:
    import tensorrt, onnx, onnxruntime
except ImportError:
    !pip install autogluon.multimodal[tests]
    !pip install -U "tensorrt>=10.0.0b0,<11.0"
```

## Dataset & Training

Using a binary classification task (PetFinder adoption speed prediction) with multimodal data (images, text, tabular):

```python
from autogluon.multimodal import MultiModalPredictor

hyperparameters = {
    "optim.max_epochs": 2,
    "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
    "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
    "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
}
predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=120,
)
```

AutoMM auto-detects problem type, data modalities, selects models, and appends a late-fusion model (MLP or transformer) on top of multiple backbones.

## TensorRT Optimization

Load and optimize the predictor:

```python
model_path = predictor.path
trt_predictor = MultiModalPredictor.load(path=model_path)
trt_predictor.optimize_for_inference()
```

`optimize_for_inference()` generates an ONNX Runtime-based module as a drop-in replacement for `torch.nn.Module`, replacing the internal `predictor._model`.

```{warning}
`optimize_for_inference()` modifies the model for inference only. Calling `predictor.fit()` afterward will error. Reload with `MultiModalPredictor.load` to refit.
```

Prediction and embedding extraction work as usual:

```python
y_pred_trt = trt_predictor.predict_proba(sample)
```

## Accuracy Considerations

FP16 mixed precision is used by default, causing minor numerical differences:

```python
np.testing.assert_allclose(y_pred, y_pred_trt, atol=0.01)
```

Evaluation metrics between PyTorch and TensorRT should be very close. If significant gaps appear, disable mixed precision:

```python
predictor.optimize_for_inference(providers=["CUDAExecutionProvider"])
```

See [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for all available providers. Refer to [TensorRT Reduced Precision Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision) for details on FP16 behavior.