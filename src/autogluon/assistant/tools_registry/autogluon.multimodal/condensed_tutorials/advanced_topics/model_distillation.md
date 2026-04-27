# Condensed: Load the Teacher Model

Summary: This tutorial demonstrates knowledge distillation using AutoGluon's `MultiModalPredictor`, compressing a 12-layer BERT teacher into a 6-layer BERT student for text classification (QNLI). It covers loading HuggingFace datasets, preparing train/valid/test splits, loading a pre-trained teacher model via `MultiModalPredictor.load()`, and performing distillation by passing `teacher_predictor` to the student's `.fit()` method. Key configurations include setting the student backbone via `model.hf_text.checkpoint_name` and training epochs via `optim.max_epochs`. Useful for implementing model compression, teacher-student training pipelines, and NLP classification with AutoGluon MultiModal.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Knowledge Distillation

## Setup & Data Preparation

```python
!pip install autogluon.multimodal
```

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("glue", "qnli")
train_valid_df = dataset["train"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=123)
test_df = dataset["validation"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
```

## Load Pre-trained Teacher Model

Teacher uses `google/bert_uncased_L-12_H-768_A-12` (12-layer BERT), student uses `google/bert_uncased_L-6_H-768_A-12` (6-layer BERT).

```python
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/unit-tests/distillation_sample_teacher.zip -O distillation_sample_teacher.zip
!unzip -q -o distillation_sample_teacher.zip -d .

from autogluon.multimodal import MultiModalPredictor
teacher_predictor = MultiModalPredictor.load("ag_distillation_sample_teacher/")
```

## Distill to Student

Pass `teacher_predictor` to `.fit()` — the student is trained by matching the teacher's prediction/feature maps, which can outperform direct finetuning.

```python
student_predictor = MultiModalPredictor(label="label")
student_predictor.fit(
    train_df,
    tuning_data=valid_df,
    teacher_predictor=teacher_predictor,
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",
        "optim.max_epochs": 2,
    }
)
print(student_predictor.evaluate(data=test_df))
```

**Key parameters:**
- `model.hf_text.checkpoint_name`: Student backbone (smaller model)
- `optim.max_epochs`: Training epochs
- `teacher_predictor`: Pre-trained teacher model passed directly to fit

For customization options and comparison with direct finetuning, see [AutoMM Distillation Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation), particularly the [multilingual distillation example](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py).