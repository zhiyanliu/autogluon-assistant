# Condensed: 2. Dataset

Summary: This tutorial demonstrates building semantic search systems using AutoGluon's MultiModalPredictor with `text_similarity` problem type and sentence-transformers. It covers three ranking approaches: BM25 baseline (using `rank_bm25` with tokenization/stopword removal), AutoMM semantic search (embedding extraction, cosine similarity ranking, NDCG evaluation via `evaluate()` and `semantic_search()` APIs), and Hybrid BM25 combining normalized BM25 scores with PLM cosine similarities using a weighted formula (β=0.3). Key techniques include `id_mappings` for query/doc text lookup, `extract_embedding()` for offline/online embedding pipelines, `compute_ranking_score` for NDCG evaluation at multiple cutoffs, and `compute_semantic_similarity` for reranking.

*This is a condensed version that preserves essential implementation details and context.*

# Semantic Search with AutoMM and Hybrid BM25

## Setup & Dataset

```python
!pip install autogluon.multimodal ir_datasets rank_bm25
```

Using the **NF Corpus (Nutrition Facts)** dataset from `ir_datasets` (323 queries, 3633 documents, 12334 relevance scores):

```python
import ir_datasets, pandas as pd
dataset = ir_datasets.load("beir/nfcorpus/test")

doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())

label_col, query_id_col, doc_id_col, text_col = "relevance", "query_id", "doc_id", "text"
id_mappings = {query_id_col: query_data.set_index(query_id_col)[text_col],
               doc_id_col: doc_data.set_index(doc_id_col)[text_col]}

# Clean data: drop URLs, concatenate title into text
query_data = query_data.drop("url", axis=1)
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)
```

## Evaluation: NDCG

**NDCG** (Normalized Discounted Cumulative Gain) penalizes relevant results appearing lower in rankings:

$$\mathrm{NDCG}_p = \frac{\mathrm{DCG}_p}{\mathrm{IDCG}_p}, \quad \mathrm{DCG}_p = \sum_{i=1}^p \frac{\mathrm{rel}_i}{\log_2(i + 1)}$$

```python
from autogluon.multimodal.utils import compute_ranking_score
cutoffs = [5, 10, 20]
```

## BM25 Baseline

BM25 parameters: **k1=1.2** (term frequency saturation), **b=0.75** (document length normalization).

```python
from collections import defaultdict
import string, nltk, numpy as np
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

def tokenize_corpus(corpus):
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    return [[w for w in nltk.word_tokenize(doc.lower()) if w not in stop_words and len(w) > 2] for doc in corpus]

def rank_documents_bm25(queries_text, queries_id, docs_id, top_k, bm25):
    tokenized_queries = tokenize_corpus(queries_text)
    results = {qid: {} for qid in queries_id}
    for qi, query in enumerate(tokenized_queries):
        scores = bm25.get_scores(query)
        for doc_idx in np.argsort(scores)[::-1][:top_k]:
            results[queries_id[qi]][docs_id[doc_idx]] = float(scores[doc_idx])
    return results

def get_qrels(dataset):
    qrel_dict = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrel_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
    return qrel_dict

qrel_dict = get_qrels(dataset)
# Evaluate BM25
tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
results = rank_documents_bm25(query_data[text_col].tolist(), query_data[query_id_col].tolist(),
                               doc_data[doc_id_col].tolist(), max(cutoffs), bm25_model)
compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
```

## AutoMM for Semantic Search

### Initialize Predictor

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    query=query_id_col, response=doc_id_col, label=label_col,
    problem_type="text_similarity",
    hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"}
)
```

### Evaluate Ranking

Automatically extracts embeddings, computes cosine similarities, ranks, and scores:

```python
predictor.evaluate(labeled_data, query_data=query_data[[query_id_col]],
    response_data=doc_data[[doc_id_col]], id_mappings=id_mappings, cutoffs=cutoffs, metrics=["ndcg"])
```

### Semantic Search & Embedding Extraction

```python
from autogluon.multimodal.utils import semantic_search
hits = semantic_search(matcher=predictor, query_data=query_data[text_col].tolist(),
    response_data=doc_data[text_col].tolist(), query_chunk_size=len(query_data), top_k=max(cutoffs))

# Extract embeddings (offline for docs, online for queries)
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
```

> **Best practice**: Extract document embeddings offline; encode only queries online. Use [Faiss](https://github.com/facebookresearch/faiss) for efficient similarity search at scale instead of `torch.topk`.

## Hybrid BM25

Combines BM25 (first-stage recall) with PLM semantic scoring for reranking:

$$score = \beta \cdot \text{normalized\_BM25} + (1 - \beta) \cdot \text{cosine\_similarity}$$

where normalized BM25 is min-max scaled. **Default: β=0.3, recall_num=1000**.

```python
import torch
from autogluon.multimodal.utils import compute_semantic_similarity

def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    bm25_scores = rank_documents_bm25(query_data[text_col].tolist(), query_data[query_id_col].tolist(),
                                       doc_data[doc_id_col].tolist(), recall_num, bm25_model)

    all_bm25 = [s for scores in bm25_scores.values() for s in scores.values()]
    max_bm25, min_bm25 = max(all_bm25), min(all_bm25)

    q_emb = dict(zip(query_data[query_id_col].tolist(), query_embeds))
    d_emb = dict(zip(doc_data[doc_id_col].tolist(), doc_embeds))

    results = {qid: {} for qid in query_data[query_id_col].tolist()}
    for qid in results:
        rec_docs = bm25_scores[qid]
        rec_doc_ids = list(rec_docs.keys())
        rec_doc_emb = torch.stack([d_emb[did] for did in rec_doc_ids])
        scores = compute_semantic_similarity(q_emb[qid], rec_doc_emb)
        scores[torch.isnan(scores)] = -1
        top_k_vals, top_k_idxs = torch.topk(scores, min(top_k + 1, len(scores[0])), dim=1, largest=True, sorted=False)

        for doc_idx, score in zip(top_k_idxs[0], top_k_vals[0]):
            did = rec_doc_ids[int(doc_idx)]
            results[qid][did] = (1 - beta) * float(score.numpy()) + \
                beta * (bm25_scores[qid][did] - min_bm25) / (max_bm25 - min_bm25)
    return results

# Evaluate
results = hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num=1000, top_k=max(cutoffs), beta=0.3)
compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
```

## Summary

| Method | Approach |
|--------|----------|
| **BM25** | Lexical matching baseline |
| **AutoMM** | Semantic embeddings via `sentence-transformers/all-MiniLM-L6-v2` — significant improvement over BM25 |
| **Hybrid BM25** | BM25 recall + PLM reranking — best results by combining lexical and semantic signals |