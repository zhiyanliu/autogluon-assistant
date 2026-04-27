# Condensed: Sentiment Analysis Task

Summary: This tutorial demonstrates using AutoGluon's `MultiModalPredictor` for NLP tasks including binary sentiment classification (SST dataset) and sentence similarity regression (STS Benchmark). It covers the complete workflow: loading data via `load_pd.load()`, training with `fit()` and `time_limit`, evaluating with multiple metrics (`acc`, `f1`, `rmse`, `pearsonr`, `spearmanr`), predicting on single inputs or batches via `predict()`/`predict_proba()`, saving/loading models, and extracting embeddings for visualization with TSNE. Key features include auto-detection of classification vs. regression from the label column, support for multiple text columns, and integration with HuggingFace Transformers, timm, and CLIP model zoos.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Text Prediction

## Setup
```python
!pip install autogluon.multimodal
```

## Sentiment Analysis (Binary Classification)

### Load Data
```python
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample for demo; use larger values for real applications
train_data = train_data.sample(n=subsample_size, random_state=0)
```
Data loads from Parquet/CSV (local or S3) into a Pandas DataFrame. Label column: **label** (0=negative, 1=positive).

### Training
```python
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label='label', eval_metric='acc', path=model_path)
predictor.fit(train_data, time_limit=180)
```
**Best practice:** Set `time_limit` much longer (e.g., 3600s) or `None` for real applications.

### Evaluation
```python
test_score = predictor.evaluate(test_data)  # uses eval_metric from init
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])  # multiple metrics
```

### Prediction
```python
predictions = predictor.predict({'sentence': [sentence1, sentence2]})
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})  # class probabilities
test_predictions = predictor.predict(test_data)  # batch prediction
```

### Save and Load

> ⚠️ **Warning:** `MultiModalPredictor.load()` uses `pickle` implicitly. Never load data from untrusted sources — arbitrary code execution is possible during unpickling.

```python
loaded_predictor = MultiModalPredictor.load(model_path)  # auto-saved after fit()
loaded_predictor.save(new_model_path)  # save to custom location
```

### Extract Embeddings
```python
embeddings = predictor.extract_embedding(test_data)  # intermediate neural network representations
```
Visualize with TSNE:
```python
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)
for val, color in [(0, 'red'), (1, 'blue')]:
    idx = (test_data['label'].to_numpy() == val).nonzero()
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=color, label=f'label={val}')
```

## Sentence Similarity (Regression)

```python
sts_train_data = load_pd.load('.../sts/train.parquet')[['sentence1', 'sentence2', 'score']]
sts_test_data = load_pd.load('.../sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
```
Label column **score** contains continuous similarity values. AutoGluon auto-detects regression.

```python
predictor_sts = MultiModalPredictor(label='score', path=sts_model_path)
predictor_sts.fit(sts_train_data, time_limit=60)

test_score = predictor_sts.evaluate(sts_test_data, metrics=['rmse', 'pearsonr', 'spearmanr'])

score = predictor_sts.predict({'sentence1': [sent_a], 'sentence2': [sent_b]}, as_pandas=False)
```

## Key Concepts

- **MultiModalPredictor** supports classification and regression; auto-detects problem type from label column
- Data tables can have **multiple text columns**
- Internally integrates **timm**, **huggingface/transformers**, and **openai/clip** as model zoo
- Unlike `TabularPredictor` (ensembles various models), `MultiModalPredictor` focuses on **selecting and finetuning deep learning models**
- See [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) and [Customize AutoMM](../advanced_topics/customization.ipynb) for more