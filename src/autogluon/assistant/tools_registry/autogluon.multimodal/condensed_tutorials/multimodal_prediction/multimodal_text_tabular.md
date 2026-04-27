# Condensed: Book Price Prediction Data

Summary: This tutorial demonstrates using AutoGluon's `MultiModalPredictor` for regression on mixed-type data (text, numeric) to predict book prices. It covers data preprocessing techniques (string parsing, log-transforming targets), training with `MultiModalPredictor` using `label` and `time_limit` parameters, and inference via `predict()`, `evaluate()`, and `extract_embedding()`. Useful for tasks involving multimodal tabular regression, automated neural network generation from heterogeneous features, and embedding extraction. Key implementation details include log-transform/inverse-transform for price targets and automatic feature type inference by AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal: Book Price Prediction

## Setup & Data

```python
!pip install autogluon.multimodal openpyxl
```

```python
import numpy as np, pandas as pd, os, warnings
from autogluon.multimodal import MultiModalPredictor
warnings.filterwarnings('ignore')
np.random.seed(123)
```

Dataset: [MachineHack Book Price Prediction](https://machinehack.com/hackathons/predict_the_price_of_books/overview) — predict book price from author, abstract, rating, etc.

```python
!mkdir -p price_of_books
!wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
!cd price_of_books && unzip -o Data.zip

train_df = pd.read_excel(os.path.join('price_of_books', 'Participants_Data', 'Data_Train.xlsx'), engine='openpyxl')
```

**Preprocessing:** Convert `Reviews`/`Ratings` to numeric, log-transform prices:

```python
def preprocess(df):
    df = df.copy(deep=True)
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)
    return df

train_df = preprocess(train_df)
train_data = train_df.iloc[100:].sample(1500, random_state=123)
test_data = train_df.iloc[:100].sample(5, random_state=245)
```

## Training

`MultiModalPredictor` auto-generates a neural network based on inferred feature types (text, numeric, etc.):

```python
import uuid
time_limit = 3 * 60  # increase for better results
model_path = f"./tmp/{uuid.uuid4().hex}-automm_text_book_price_prediction"
predictor = MultiModalPredictor(label='Price', path=model_path)
predictor.fit(train_data, time_limit=time_limit)
```

## Prediction & Evaluation

```python
predictions = predictor.predict(test_data)
print(np.exp(predictions) - 1)  # inverse log-transform

performance = predictor.evaluate(test_data)

embeddings = predictor.extract_embedding(test_data)  # extract data embeddings
embeddings.shape
```

**Key capabilities:** `predict()`, `evaluate()`, and `extract_embedding()` all work directly on the test DataFrame.

**Resources:** [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) | [Customize AutoMM](../advanced_topics/customization.ipynb)