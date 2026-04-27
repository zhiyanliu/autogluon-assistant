# Condensed: Training

Summary: This tutorial provides a complete `MultilabelPredictor` class implementation for AutoGluon that predicts multiple target columns simultaneously by wrapping multiple `TabularPredictor` instances. It covers building a multi-output prediction system supporting mixed problem types (regression, multiclass, binary) with optional auto-regressive label correlation, where earlier predictions feed as features to later ones. Key techniques include per-label predictor management, selective column dropping during training, pickle-based save/load persistence, and chained prediction during inference. The tutorial helps with tasks involving multi-label tabular prediction, model serialization, evaluation across heterogeneous targets, and accessing individual sub-predictors for single-label use.

*This is a condensed version that preserves essential implementation details and context.*

# MultilabelPredictor with AutoGluon

## Installation
```python
!pip install autogluon.tabular[all]
```

## MultilabelPredictor Implementation

A wrapper that creates separate `TabularPredictor` instances for each label column, with optional auto-regressive label correlation.

**Key parameters:**
- `labels`: List of columns to predict
- `consider_labels_correlation`: If `True`, predictions are made auto-regressively (earlier labels fed as features to later ones); label ordering matters
- `problem_types` / `eval_metrics`: Per-label problem type and metric lists (optional)

```python
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path

class MultilabelPredictor:
    multi_predictor_file = 'multilabel_predictor.pkl'

    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=True, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels")
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}
        self.eval_metrics = {labels[i]: eval_metrics[i] for i in range(len(labels))} if eval_metrics else {}
        for i, label in enumerate(labels):
            path_i = os.path.join(self.path, "Predictor_" + str(label))
            self.predictors[label] = TabularPredictor(
                label=label,
                problem_type=problem_types[i] if problem_types else None,
                eval_metric=eval_metrics[i] if eval_metrics else None,
                path=path_i, **kwargs)

    def fit(self, train_data, tuning_data=None, **kwargs):
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        tuning_data_og = tuning_data.copy() if tuning_data is not None else None
        save_metrics = len(self.eval_metrics) == 0
        for i, label in enumerate(self.labels):
            predictor = self.get_predictor(label)
            # Drop future labels (correlation) or all other labels (independent)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=os.path.join(self.path, self.multi_predictor_file), object=self)

    @classmethod
    def load(cls, path):
        path = os.path.expanduser(path)
        return load_pkl.load(path=os.path.join(path, cls.multi_predictor_file))

    def get_predictor(self, label):
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(path=predictor)
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass=True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)  # feeds predictions to subsequent predictors
        return data[self.labels] if not as_proba else predproba_dict
```

## Training

```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=500, random_state=0)

labels = ['education-num', 'education', 'class']
problem_types = ['regression', 'multiclass', 'binary']
eval_metrics = ['mean_absolute_error', 'accuracy', 'accuracy']

multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types,
                                       eval_metrics=eval_metrics, path='agModels-predictEducationClass')
multi_predictor.fit(train_data, time_limit=5)
```

## Inference & Evaluation

```python
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
multi_predictor = MultilabelPredictor.load('agModels-predictEducationClass')

predictions = multi_predictor.predict(test_data.drop(columns=labels))
evaluations = multi_predictor.evaluate(test_data)
```

## Accessing Individual Predictors

```python
predictor_class = multi_predictor.get_predictor('class')
predictor_class.leaderboard()
```

> **Warning:** Set `consider_labels_correlation=False` if you plan to use individual predictors independently, since correlated predictors expect prior label predictions as input features.

## Best Practices

- **Specify `eval_metrics`** matching your actual evaluation criteria
- **Use `presets='best_quality'`** for best performance (enables stack ensembling)
- **Memory/disk issues:** See "In Depth Tutorial" for mitigation strategies
- **Slow inference:** Use `presets=['good_quality', 'optimize_for_deployment']`