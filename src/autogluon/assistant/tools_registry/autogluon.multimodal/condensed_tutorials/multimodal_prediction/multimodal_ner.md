# Condensed: Get the Twitter Dataset

Summary: This tutorial demonstrates multimodal Named Entity Recognition (NER) using AutoGluon's `MultiModalPredictor` with combined text and image inputs. It covers data preparation (loading CSVs, expanding image paths to absolute paths), training with `problem_type="ner"` and `column_types={"text_snippet": "text_ner"}` to specify the NER target column, evaluation using precision/recall/F1 metrics, and prediction parsing (entity dicts with `start`, `end`, `entity_group`). It also shows model saving/reloading via `MultiModalPredictor.load()` and continuous training. Useful for implementing multimodal NER pipelines, handling multi-column datasets, and leveraging AutoGluon's automatic modality detection and late-fusion.

*This is a condensed version that preserves essential implementation details and context.*

# Multimodal Named Entity Recognition with AutoGluon

## Setup & Data Loading

```python
!pip install autogluon.multimodal
```

Uses the [Twitter dataset](https://github.com/jefferyYu/UMT/tree/master) (2016-2017 tweets with text + images).

```python
from autogluon.core.utils.loaders import load_zip
download_dir = './ag_automm_tutorial_ner'
load_zip.unzip('https://automl-mm-bench.s3.amazonaws.com/ner/multimodal_ner.zip', unzip_dir=download_dir)

dataset_path = download_dir + '/multimodal_ner'
train_data = pd.read_csv(f'{dataset_path}/twitter17_train.csv')
test_data = pd.read_csv(f'{dataset_path}/twitter17_test.csv')
label_col = 'entity_annotations'
```

**Expand image paths** to absolute paths for training:

```python
image_col = 'image'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])  # Use first image only
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(base_folder + path) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

Each row contains: text, image path, and `entity_annotations` (NER labels for the text column).

## Training

**Key parameters:**
- `problem_type="ner"` — required for NER tasks
- `column_types={"text_snippet": "text_ner"}` — **critical when multiple text columns exist**, identifies which column to extract entities from

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner"
predictor = MultiModalPredictor(problem_type="ner", label="entity_annotations", path=model_path)
predictor.fit(
    train_data=train_data,
    column_types={"text_snippet": "text_ner"},
    time_limit=300,  # seconds
)
```

AutoMM auto-detects modalities, selects models from multimodal pools, and applies late-fusion when multiple backbones are used.

## Evaluation

```python
predictor.evaluate(test_data, metrics=['overall_recall', 'overall_precision', 'overall_f1'])
```

## Prediction

```python
prediction_input = test_data.drop(columns=label_col).head(1)
predictions = predictor.predict(prediction_input)

for entity in predictions[0]:
    print(f"Word '{prediction_input.text_snippet[0][entity['start']:entity['end']]}' belongs to group: {entity['entity_group']}")
```

Predictions return entity dicts with `start`, `end`, and `entity_group` keys.

## Reloading & Continuous Training

```python
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1'])
```

Models are auto-saved and can be reloaded for continued training with new data or additional time budget.