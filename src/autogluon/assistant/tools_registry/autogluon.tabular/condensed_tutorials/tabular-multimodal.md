# Condensed: Preparing the image column

Summary: This tutorial demonstrates how to use AutoGluon's TabularPredictor for multimodal learning combining tabular, text, and image features simultaneously. Key techniques include: preprocessing image columns to handle single images per row with absolute path expansion, configuring `FeatureMetadata` to manually register image columns via `add_special_types({image_col: ['image_path']})`, using the built-in `'multimodal'` hyperparameter config (`get_hyperparameter_config('multimodal')`) which trains tabular models plus BERT and ResNet, and fitting with `time_limit` control. Useful for implementing multi-class classification pipelines that jointly leverage mixed data modalities through AutoGluon's unified API.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Multimodal Tutorial (Tabular + Text + Image)

## Setup & Data Loading

```python
!pip install autogluon

from autogluon.core.utils.loaders import load_zip
download_dir = './ag_petfinder_tutorial'
load_zip.unzip('https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip', unzip_dir=download_dir)

import pandas as pd
dataset_path = download_dir + '/petfinder_processed'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)  # dev has ground truth labels

label = 'AdoptionSpeed'  # Multi-class classification (5 categories)
image_col = 'Images'
```

## Image Column Preprocessing

**Important:** AutoGluon supports only one image per row. Extract the first image and expand to absolute paths:

```python
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

import os
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

**Best Practice:** Sample data when prototyping — multimodal training is computationally intensive, especially with `best_quality` preset.

```python
train_data = train_data.sample(500, random_state=0)
```

## FeatureMetadata Configuration

AutoGluon auto-detects text columns, but **image columns must be manually specified** via `image_path` special type:

```python
from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
```

## Hyperparameters

Use the built-in `'multimodal'` config (trains tabular models + Electra BERT text model + ResNet image model):

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

## Training & Evaluation

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,
)

leaderboard = predictor.leaderboard(test_data)
```

**Key points:**
- Pass `feature_metadata` with image special type to enable image features
- `time_limit=900` (15 min) controls training duration
- The predictor leverages tabular, text, and image features simultaneously