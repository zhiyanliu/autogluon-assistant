# Condensed: Multiple Label Columns with AutoMM

Summary: This tutorial provides workarounds for handling multiple label columns with AutoGluon MultiModalPredictor, which doesn't natively support multi-label output. It covers two scenarios: (1) mutually exclusive labels—combining multiple binary columns into a single categorical column, training one predictor, and converting predictions back; (2) non-mutually exclusive labels (multi-label classification)—training separate predictors per label while excluding other label columns from features. Key techniques include label column merging/splitting, iterative model training, proper feature isolation, and time budget allocation across multiple predictors. Useful for multi-label classification and multi-output prediction tasks with AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

# Multiple Label Columns with AutoMM

AutoGluon MultiModal doesn't natively support multiple label columns. Two approaches based on label relationships:

## Mutually Exclusive Labels

Combine into a single column:

```python
def combine_labels(row, label_columns):
    for label in label_columns:
        if row[label] == 1:
            return label
    return 'none'

df['combined_label'] = df.apply(lambda row: combine_labels(row, label_columns), axis=1)

predictor = MultiModalPredictor(label='combined_label').fit(df)

# Convert predictions back to multiple columns
predictions = predictor.predict(test_data)
for label in label_columns:
    test_data[f'{label}'] = (predictions == label).astype(int)
```

## Non-Mutually Exclusive Labels

Train separate predictors per label, **excluding other label columns from features**:

```python
label_columns = ['label1', 'label2', 'label3']
predictors = {}

for label in label_columns:
    train_df = df.drop(columns=[l for l in label_columns if l != label])
    predictors[label] = MultiModalPredictor(label=label).fit(train_df)

for label in label_columns:
    test_features = test_data.drop(columns=label_columns)
    test_data[f'pred_{label}'] = predictors[label].predict(test_features)
```

**Important:** With N label columns, allocate `total_time / N` as `time_limit` for each predictor.