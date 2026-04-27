# Condensed: Zero Shot Evaluation

Summary: This tutorial demonstrates semantic segmentation using AutoGluon's MultiModalPredictor with SAM (Segment Anything Model) fine-tuning via LoRA. It covers: loading segmentation datasets with image/mask path pairs, expanding relative paths to absolute using a path_expander utility, zero-shot SAM evaluation with `problem_type="semantic_segmentation"` and `num_classes=1`, fine-tuning SAM using `predictor.fit()` with `time_limit` and `tuning_data`, evaluating with IoU metrics, generating predictions, and saving/loading predictors. Key configurations include `model.sam.checkpoint_name` for selecting SAM variants (e.g., `facebook/sam-vit-base`) and the automatic LoRA-based efficient fine-tuning approach.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Semantic Segmentation with SAM Fine-tuning

## Setup & Data Loading

```python
!pip install autogluon.multimodal
```

```python
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/leaf_disease_segmentation.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Load CSVs and **expand relative paths to absolute** for correct data loading:

```python
import pandas as pd, os
dataset_path = os.path.join(download_dir, 'leaf_disease_segmentation')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col, label_col = 'image', 'label'

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[per_col] = val_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[per_col] = test_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

DataFrames contain two columns: image paths and corresponding groundtruth mask paths.

## Zero-Shot Evaluation

```python
from autogluon.multimodal import MultiModalPredictor
predictor_zero_shot = MultiModalPredictor(
    problem_type="semantic_segmentation",
    label=label_col,
    hyperparameters={"model.sam.checkpoint_name": "facebook/sam-vit-base"},
    num_classes=1,  # foreground-background segmentation
)
pred_zero_shot = predictor_zero_shot.predict({'image': [test_data.iloc[0]['image']]})
scores = predictor_zero_shot.evaluate(test_data, metrics=["iou"])
```

**Key insight:** SAM without prompts outputs a rough leaf mask rather than disease masks—it lacks domain context. While click prompts help, it's not ideal for standalone deployment.

## Fine-tune SAM

```python
import uuid
save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    label="label",
    hyperparameters={"model.sam.checkpoint_name": "facebook/sam-vit-base"},
    path=save_path,
)
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180,  # seconds
)
```

**Implementation detail:** Fine-tuning uses [LoRA](https://arxiv.org/abs/2106.09685) for efficiency. Without hyperparameter customization, the huge SAM is the default model, making efficient fine-tuning essential.

```python
scores = predictor.evaluate(test_data, metrics=["iou"])
pred = predictor.predict({'image': [test_data.iloc[0]['image']]})
```

Fine-tuning significantly improves test IoU scores and produces masks much closer to groundtruth.

## Save and Load

The predictor auto-saves after `fit()`. Reload with:

```python
loaded_predictor = MultiModalPredictor.load(save_path)
scores = loaded_predictor.evaluate(test_data, metrics=["iou"])
```

> ⚠️ **Warning:** `MultiModalPredictor.load()` uses `pickle` implicitly, which is insecure. Malicious pickle data can execute arbitrary code during unpickling. **Only load data you trust.**

## References

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)