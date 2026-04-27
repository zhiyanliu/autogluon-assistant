# Condensed: Training

Summary: This tutorial demonstrates Chinese Named Entity Recognition (NER) using AutoGluon's MultiModalPredictor. It covers loading CSV-formatted NER datasets with `load_pd`, configuring `MultiModalPredictor` with `problem_type="ner"`, training with a Chinese-specific backbone (`hfl/chinese-lert-small`) via the `model.ner_text.checkpoint_name` hyperparameter, setting `time_limit` for training, evaluating on dev data, predicting entities on both dataset rows and custom string inputs (passed as dict with `text_snippet` key), and visualizing NER results with `visualize_ner`. Key insight: Chinese NER uses the identical API as English NER—only the pretrained checkpoint changes.

*This is a condensed version that preserves essential implementation details and context.*

# Chinese NER with AutoGluon MultiModal

## Setup & Data

```python
!pip install autogluon.multimodal
```

```python
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal.utils import visualize_ner

train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_train.csv')
dev_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_dev.csv')
```

Entity labels: **HPPX** (brand), **HCCX** (product), **XH** (pattern), **MISC** (miscellaneous/specs).

```python
visualize_ner(train_data["text_snippet"].iloc[0], train_data["entity_annotations"].iloc[0])
```

## Training

**Key:** Chinese NER follows the same process as English NER — just select a Chinese/multilingual pretrained checkpoint. Here we use `hfl/chinese-lert-small`.

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'hfl/chinese-lert-small'},
    time_limit=300,
)
```

## Evaluation & Prediction

```python
predictor.evaluate(dev_data)

output = predictor.predict(dev_data)
visualize_ner(dev_data["text_snippet"].iloc[0], output[0])
```

Predict on custom input:

```python
sentence = "2023年兔年挂件新年装饰品小挂饰乔迁之喜门挂小兔子"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

## Further Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)