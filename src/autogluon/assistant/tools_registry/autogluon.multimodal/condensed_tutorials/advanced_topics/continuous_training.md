# Condensed: Use Case 1: Expanding Training with Additional Data or Training Time

Summary: This tutorial demonstrates continuous training with AutoGluon's MultiModalPredictor across three use cases: (1) extending training by loading a saved predictor via `MultiModalPredictor.load(model_path)` and calling `.fit()` again with new data, (2) resuming crashed training using `MultiModalPredictor.load(path, resume=True)`, and (3) transferring learned weights to new tasks via `predictor.dump_model()` and specifying custom checkpoints through `hyperparameters` (e.g., `model.hf_text.checkpoint_name`, `model.timm_image.checkpoint_name`, `model.mmdet_image.checkpoint_name`). It covers incremental data addition, checkpoint management (`model.ckpt` vs `last.ckpt`), cross-task fine-tuning, and warns about catastrophic forgetting.

*This is a condensed version that preserves essential implementation details and context.*

# Continuous Training with AutoMM

AutoMM supports three continuous training use cases: extending training, resuming interrupted training, and transferring to new tasks.

## Use Case 1: Extending Training with More Data/Time

Continue training to address underfitting or incorporate new data (same problem type/classes for multiclass).

**Initial training** on SST binary sentiment dataset:

```python
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal import MultiModalPredictor
import uuid

train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
train_data_1 = train_data.sample(n=1000, random_state=0)

model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)
test_score = predictor.evaluate(test_data)
```

**Continue training** — load the saved predictor and call `.fit()` again with same or new data:

```python
predictor_2 = MultiModalPredictor.load(model_path)
train_data_2 = train_data.drop(train_data_1.index).sample(n=1000, random_state=0)
predictor_2.fit(train_data_2, time_limit=60)
```

> **Note:** `model.ckpt` is saved under `model_path` after successful training. Set longer `time_limit` (e.g., 3600) or `time_limit=None` for real applications.

## Use Case 2: Resuming from Last Checkpoint

If training crashes, `last.ckpt` (not `model.ckpt`) is saved. Resume with:

```python
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

## Use Case 3: Transfer to New Tasks

Dump a trained model's weights and use them as a foundation for a different task.

**Dump model:**
```python
dump_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor.dump_model(save_path=dump_model_path)
```

**Fine-tune on new task** (e.g., STS regression using the binary sentiment model):

```python
sts_train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet")[
    ["sentence1", "sentence2", "score"]
]
sts_test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet")[
    ["sentence1", "sentence2", "score"]
]

sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label="score", path=sts_model_path)
predictor_sts.fit(
    sts_train_data,
    hyperparameters={"model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"},
    time_limit=30
)
test_score = predictor_sts.evaluate(sts_test_data, metrics=["rmse", "pearsonr", "spearmanr"])
```

**Supported custom model hyperparameters:**

| Model Type | Hyperparameter Key |
|---|---|
| HuggingFace text | `model.hf_text.checkpoint_name` |
| TIMM image | `model.timm_image.checkpoint_name` |
| MMDetection | `model.mmdet_image.checkpoint_name` |

Fusion models comprising these are also supported.

> **Warning:** [Catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference) is a significant challenge when transferring to new tasks — the model may lose previously learned knowledge.