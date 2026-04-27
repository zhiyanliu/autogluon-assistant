# Condensed: Installation

Summary: This tutorial demonstrates using AutoGluon's tabular foundation models—Mitra, TabICL, and TabPFNv2—for classification and regression tasks. It covers installation via `autogluon.tabular[mitra|tabicl|tabpfn]`, configuring each model through `hyperparameters` in `TabularPredictor.fit()`, Mitra's zero-shot vs. fine-tuning modes (`fine_tune`, `fine_tune_steps`), TabICL for large datasets, TabPFNv2 for small datasets (<10K samples), and ensembling multiple foundation models in a single predictor. Key APIs include `predict()`, `predict_proba()`, and `leaderboard()`. Useful for implementing tabular foundation models with minimal code and combining them for enhanced performance.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Foundational Models Tutorial (Condensed)

## Installation

```python
!pip install uv
!uv pip install autogluon.tabular[mitra]    # Mitra
!uv pip install autogluon.tabular[tabicl]   # TabICL
!uv pip install autogluon.tabular[tabpfn]   # TabPFNv2
```

```python
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, fetch_california_housing
```

## Data Preparation

```python
# Load datasets
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df['target'] = housing_data.target

# 80/20 splits
wine_train, wine_test = train_test_split(wine_df, test_size=0.2, random_state=42, stratify=wine_df['target'])
housing_train, housing_test = train_test_split(housing_df, test_size=0.2, random_state=42)

# Convert to TabularDataset
wine_train_data, wine_test_data = TabularDataset(wine_train), TabularDataset(wine_test)
housing_train_data, housing_test_data = TabularDataset(housing_train), TabularDataset(housing_test)
```

## 1. Mitra

State-of-the-art tabular foundation model by the AutoGluon team. Pretrained on synthetic data, excels on **small datasets (<5,000 samples, <100 features)** for classification and regression. Supports **zero-shot and fine-tuning**, runs on GPU/CPU. Apache-2.0 licensed.

**Classification (zero-shot):**
```python
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(wine_train_data, hyperparameters={'MITRA': {'fine_tune': False}})

mitra_predictor.predict(wine_test_data)
mitra_predictor.predict_proba(wine_test_data)
mitra_predictor.leaderboard(wine_test_data)
```

**Classification (fine-tuned):**
```python
mitra_predictor_ft = TabularPredictor(label='target')
mitra_predictor_ft.fit(
    wine_train_data,
    hyperparameters={'MITRA': {'fine_tune': True, 'fine_tune_steps': 10}},
    time_limit=120,
)
```

**Regression:**
```python
mitra_reg_predictor = TabularPredictor(label='target', problem_type='regression')
mitra_reg_predictor.fit(
    housing_train_data.sample(1000),
    hyperparameters={'MITRA': {'fine_tune': False}},
)
```

## 2. TabICL

Transformer-based in-context learning model for **large tabular datasets**. Effective with limited training data. ([Paper](https://arxiv.org/abs/2502.05564))

```python
tabicl_predictor = TabularPredictor(label='target')
tabicl_predictor.fit(wine_train_data, hyperparameters={'TABICL': {}})
tabicl_predictor.leaderboard(wine_test_data)
```

## 3. TabPFNv2

Prior-fitted networks optimized for **small datasets (<10,000 samples)**. Works best with default parameters. ([Paper](https://www.nature.com/articles/s41586-024-08328-6))

```python
tabpfnv2_predictor = TabularPredictor(label='target')
tabpfnv2_predictor.fit(wine_train_data, hyperparameters={'TABPFNV2': {}})
tabpfnv2_predictor.leaderboard(wine_test_data)
```

## Combining Multiple Foundational Models

AutoGluon supports stacking/ensembling multiple foundation models in a single predictor:

```python
ensemble_predictor = TabularPredictor(label='target').fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': True, 'fine_tune_steps': 10},
        'TABPFNV2': {},
        'TABICL': {},
    },
    time_limit=300,
)
ensemble_predictor.leaderboard(wine_test_data)
```