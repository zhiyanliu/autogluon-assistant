# Condensed: Finetune the German BERT

Summary: This tutorial demonstrates cross-lingual text classification using AutoGluon's `MultiModalPredictor`, covering two approaches: finetuning a language-specific model (German BERT via `model.hf_text.checkpoint_name`) and enabling zero-shot cross-lingual transfer with `presets='multilingual'` (DeBERTa-V3 backbone). It shows how to train on one language (English) and evaluate on others (German, Japanese) without translation. Key implementation details include specifying HuggingFace checkpoints, configuring training epochs via hyperparameters, loading multilingual TSV datasets, and using `.fit()` and `.evaluate()` APIs. Useful for building multilingual sentiment classifiers and implementing zero-shot cross-lingual NLP pipelines.

*This is a condensed version that preserves essential implementation details and context.*

# Cross-Lingual Text Classification with AutoGluon MultiModal

## Setup & Data

```python
!pip install autogluon.multimodal
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```

Load German and English train/test splits (TSV format with `label` and `text` columns):

```python
import pandas as pd
train_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_train.tsv',
                          sep='\t', header=None, names=['label', 'text']).sample(1000, random_state=123)
test_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_test.tsv',
                          sep='\t', header=None, names=['label', 'text']).sample(200, random_state=123)
train_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_train.tsv',
                          sep='\t', header=None, names=['label', 'text']).sample(1000, random_state=123)
test_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_test.tsv',
                          sep='\t', header=None, names=['label', 'text']).sample(200, random_state=123)
```

## Approach 1: Language-Specific Model (German BERT)

Finetune a language-specific model via HuggingFace checkpoint name:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label')
predictor.fit(train_de_df,
              hyperparameters={
                  'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                  'optim.max_epochs': 2
              })
```

**Result:** Good performance on German, poor on English — no cross-lingual capability.

## Approach 2: Cross-Lingual Transfer with `presets='multilingual'`

Use `presets='multilingual'` to load a multilingual backbone (internally uses DeBERTa-V3). Train on **English only**, then evaluate on any language (zero-shot transfer):

```python
predictor = MultiModalPredictor(label='label')
predictor.fit(train_en_df,
              presets='multilingual',
              hyperparameters={
                  'optim.max_epochs': 2
              })
```

```python
score_in_en = predictor.evaluate(test_en_df)  # Works for English
score_in_de = predictor.evaluate(test_de_df)  # Works for German (zero-shot)
```

Test on Japanese (unseen language during training):

```python
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']).sample(200, random_state=123)
score_in_jp = predictor.evaluate(test_jp_df)  # Also works for Japanese
```

## Key Concepts

- **`model.hf_text.checkpoint_name`**: Specify any HuggingFace model checkpoint for text backbone
- **`presets='multilingual'`**: Enables cross-lingual transfer using multilingual pretrained models (DeBERTa-V3)
- **Zero-shot cross-lingual transfer**: Train on one language, evaluate on others without translation — based on [XLM-R](https://arxiv.org/pdf/1911.02116.pdf) approach
- This outperforms the "translate-test" baseline (translating target language to source language before inference)