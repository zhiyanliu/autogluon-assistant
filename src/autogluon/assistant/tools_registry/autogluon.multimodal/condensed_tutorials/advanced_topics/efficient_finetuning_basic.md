# Condensed: Finetuning Multilingual Model with IA3 + BitFit

Summary: This tutorial demonstrates parameter-efficient finetuning (PEFT) using AutoGluon's MultiModalPredictor with IA3+BitFit (`optim.peft: "ia3_bias"`), tuning only ~0.5% of parameters for multilingual sentiment classification. It covers configuring hyperparameters for efficient training, cross-lingual evaluation (English/German/Japanese), and finetuning large models like FLAN-T5-XL (~1.2B params) on a single T4 GPU using gradient checkpointing (`model.hf_text.gradient_checkpointing: True`) combined with PEFT. Key tasks include setting up MultiModalPredictor with multilingual presets, configuring learning rate/batch size/epochs, memory optimization via `low_cpu_mem_usage`, and evaluating cross-lingual transfer performance.

*This is a condensed version that preserves essential implementation details and context.*

# Efficient Finetuning with AutoGluon MultiModal (IA3 + BitFit)

## Setup & Data

```python
!pip install autogluon.multimodal
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```

```python
import pandas as pd

train_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_train.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
                .sample(1000, random_state=123).reset_index(drop=True)

test_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_test.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)
# Similarly load de_test.tsv and jp_test.tsv
```

## Finetuning Multilingual Model with IA3 + BitFit

Enable efficient finetuning by setting `optim.peft` to `"ia3_bias"` — tunes only **~0.5% of parameters** with results comparable to full finetuning.

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(train_en_df,
              presets="multilingual",
              hyperparameters={
                  "optim.peft": "ia3_bias",
                  "optim.lr_decay": 0.9,
                  "optim.lr": 3e-03,
                  "optim.end_lr": 3e-03,
                  "optim.max_epochs": 2,
                  "optim.warmup_steps": 0,
                  "env.batch_size": 32,
              })
```

**Key result:** English-only training achieves good cross-lingual performance on German and Japanese test sets.

```python
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)
score_in_jp = predictor.evaluate(test_jp_df)
```

## Training FLAN-T5-XL (~1.2B params) on Single GPU

Combine **gradient checkpointing** + **PEFT** to finetune large models on a single T4 GPU. Set `"model.hf_text.gradient_checkpointing": True`.

```python
predictor = MultiModalPredictor(label="label", path=new_model_path)
predictor.fit(train_en_df.sample(200, random_state=123),
              presets="multilingual",
              hyperparameters={
                  "model.hf_text.checkpoint_name": "google/flan-t5-xl",
                  "model.hf_text.gradient_checkpointing": True,
                  "model.hf_text.low_cpu_mem_usage": True,
                  "optim.peft": "ia3_bias",
                  "optim.lr_decay": 0.9,
                  "optim.lr": 3e-03,
                  "optim.end_lr": 3e-03,
                  "optim.max_epochs": 1,
                  "optim.warmup_steps": 0,
                  "env.batch_size": 1,
                  "env.inference_batch_size_ratio": 1
              })
```

**Results:** 1.2B total params, only **203K trainable**. Achieves `roc_auc: 0.931` on English test set with just 200 training samples and 1 epoch.

### Critical Parameters

| Parameter | Purpose |
|---|---|
| `optim.peft: "ia3_bias"` | Enables IA3 + BitFit efficient finetuning |
| `model.hf_text.gradient_checkpointing: True` | Reduces GPU memory for large models |
| `model.hf_text.low_cpu_mem_usage: True` | Reduces CPU memory during model loading |
| `env.batch_size: 1` | Required for large models on limited GPU memory |
| `presets: "multilingual"` | Uses multilingual backbone model |