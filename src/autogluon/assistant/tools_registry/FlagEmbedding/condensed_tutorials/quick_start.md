# Condensed: Quick Start

Summary: This tutorial demonstrates building a semantic text retrieval pipeline using FlagEmbedding's BGE model (`BAAI/bge-base-en-v1.5`). It covers initializing `FlagModel` with `query_instruction_for_retrieval` and `use_fp16` parameters, encoding queries and corpus into 768-dim embeddings via `model.encode()`, computing dot product similarity for ranking, and evaluating retrieval quality using a Mean Reciprocal Rank (MRR) implementation with configurable cutoffs. Useful for tasks involving semantic search, document retrieval, embedding-based similarity ranking, and retrieval evaluation metrics.

*This is a condensed version that preserves essential implementation details and context.*

# Quick Start: BGE Models for Text Retrieval

## Setup

```python
%pip install -U FlagEmbedding
```

```python
corpus = [
    "Michael Jackson was a legendary pop icon known for his record-breaking music and dance innovations.",
    "Fei-Fei Li is a professor in Stanford University, revolutionized computer vision with the ImageNet project.",
    "Brad Pitt is a versatile actor and producer known for his roles in films like 'Fight Club' and 'Once Upon a Time in Hollywood.'",
    "Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.",
    "Eminem is a renowned rapper and one of the best-selling music artists of all time.",
    "Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.",
    "Sam Altman leads OpenAI as its CEO, with astonishing works of GPT series and pursuing safe and beneficial AI.",
    "Morgan Freeman is an acclaimed actor famous for his distinctive voice and diverse roles.",
    "Andrew Ng spread AI knowledge globally via public courses on Coursera and Stanford University.",
    "Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.",
]
query = "Who could be an expert of neural network?"
```

## Step 1: Text → Embedding

```python
from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

corpus_embeddings = model.encode(corpus)   # shape: (10, 768)
query_embedding = model.encode(query)      # shape: (768,)
```

## Step 2: Similarity & Ranking

Compute dot product similarity, then rank by descending score:

```python
sim_scores = query_embedding @ corpus_embeddings.T

sorted_indices = sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True)
```

**Key insight:** The model correctly identifies Geoffrey Hinton (#3) and Fei-Fei Li (#1) as top results despite "neural network" not appearing in any corpus sentence — demonstrating strong semantic understanding.

## Step 3: Evaluate with MRR

```python
queries = [
    "Who could be an expert of neural network?",
    "Who might had won Grammy?",
    "Won Academy Awards",
    "One of the most famous female singers.",
    "Inventor of AlexNet",
]
ground_truth = [[1, 3], [0, 4, 5], [2, 7, 9], [5], [3]]

queries_embedding = model.encode(queries)
scores = queries_embedding @ corpus_embeddings.T
rankings = [sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True) for sim_scores in scores]
```

MRR implementation:

```python
def MRR(preds, labels, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, label in zip(preds, labels):
        for i, c in enumerate(cutoffs):
            for j, index in enumerate(pred):
                if j < c and index in label:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr

cutoffs = [1, 5]
mrrs = MRR(rankings, ground_truth, cutoffs)
# Results: MRR@1: 0.8, MRR@5: 0.9
```

**Key parameters:** `query_instruction_for_retrieval` prepends an instruction to queries (not corpus); `use_fp16=True` enables half-precision for efficiency. Embeddings are 768-dimensional vectors. Similarity is computed via dot product.