# Enterprise Semantic Search Engine

An end-to-end enterprise semantic search system built with Hugging Face Transformers and fine-tuned Large Language Models.

The goal of this project is to build an enterprise-oriented semantic search system that takes a customer's query, retrieves relevant information, and generates an appropriate answer. The project focuses on the complete LLM workflow — from dataset analysis and model benchmarking to fine-tuning, retrieval, vector search, RAG, evaluation, and qualitative error analysis.

---

## 📌 Project Overview

This project explores the development of an enterprise-oriented semantic search and Retrieval-Augmented Generation (RAG) system.

The project starts with sentence embeddings and dense retrieval, then progresses through vector search, retrieval evaluation, reranking, context construction, prompt construction, and LLM-based answer generation.

The overall pipeline is:

```text
Documents
    ↓
Sentence Embeddings
    ↓
Dense Retrieval
    ↓
Vector Search
    ↓
Reranking
    ↓
Context Construction
    ↓
Prompt Construction
    ↓
LLM Generation
    ↓
RAG Evaluation
```

The project follows an experimental approach in which different methods are benchmarked using standard information retrieval metrics. The goal is not only to build a working semantic search system, but also to analyze retrieval quality, search efficiency, reranking effectiveness, context selection, and the grounding and relevance of generated answers.

---

## 🔄 Workflow

The project follows an end-to-end pipeline from raw documents and queries to semantic retrieval and answer generation.

```text
Documents + Queries
        ↓
Exploratory Data Analysis
        ↓
Sentence Embedding Benchmarking
        ↓
Embedding Model Fine-Tuning
        ↓
Dense Retrieval
        ↓
Retrieval Evaluation
        ↓
Vector Search
        ↓
Reranking
        ↓
Context Construction
        ↓
Prompt Construction
        ↓
LLM Generation
        ↓
RAG Evaluation
```

Each stage is evaluated independently to identify how different components affect the overall system performance.
---
## 📂 Project Structure


```text
Enterprise_Semantic_Search_Engine/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ModelBenchmarking.ipynb
│   └── 03_RAG_pipline.ipynb
│
├── Src/
│   ├── preprocessing.py
│   ├── vectorSearch.py
│   ├── Benchmark.py
│   ├── embed.py
│   ├── evaluate.py
│   ├── retrieve.py
│   ├── train.py
│   │
│   └── RAG/
│       ├── context.py
│       ├── prompting.py
│       ├── generation.py
│       └── evaluation.py
│
├── README.md
└── requirements.txt
```

### Directory Overview

* **`notebooks/`** — Exploratory analysis, experiments, and model benchmarking.
* **`Src/`** — Reusable components for preprocessing, embedding, training, retrieval, vector search, and evaluation.
* **`Src/RAG/`** — Components of the Retrieval-Augmented Generation pipeline, including context construction, prompt construction, and LLM generation.
* **`README.md`** — Project documentation, methodology, experiments, and results.
* **`requirements.txt`** — Python dependencies required to reproduce the project.

---

## 📊 Dataset & Preprocessing

The project uses a corpus of **25,657 documents** and **1,000 queries** for semantic retrieval.

The dataset follows an asymmetric **query-to-document retrieval** setting, where each query is associated with one or more relevant documents through relevance judgments (qrels).

### Dataset Statistics

| Component               |   Size |
| ----------------------- | -----: |
| Documents               | 25,657 |
| Queries                 |  1,000 |
| Relevance Pairs (Qrels) | 29,928 |
| Embedding Dimension     |    384 |

The queries and documents are used throughout the project to benchmark sentence embedding models, train the retrieval model, evaluate dense retrieval, and build the subsequent vector search and RAG pipeline.

For retrieval experiments, the queries were divided into:

| Split      | Queries |
| ---------- | ------: |
| Train      |     800 |
| Validation |     100 |
| Test       |     100 |

The relevance judgments were kept separate from the retrieval pipeline and were used only for offline evaluation.

---

## 🔍 Exploratory Data Analysis

The initial exploratory analysis was performed to understand the structure of the retrieval dataset, the distribution of queries and documents, and the relevance relationships between them.

### Dataset Overview

The corpus contains:

* **25,657 documents**
* **1,000 queries**
* **29,928 query-document relevance pairs**

Each query can be associated with multiple relevant documents, making this an asymmetric query-to-document retrieval task rather than a one-query-one-answer classification problem.

### Retrieval Data Distribution

The relevance judgments contain approximately **29.9 relevant documents per query on average**.

The queries were divided into:

| Split      | Queries |
| ---------- | ------: |
| Train      |     800 |
| Validation |     100 |
| Test       |     100 |

The corresponding relevance judgments were kept separate for training and evaluation.

### Document Length Analysis

The document corpus contains relatively long documents. This observation became important later when designing the RAG context.

For the documents retrieved in the final Top-10 analysis:

| Statistic | Word Count |
| --------- | ---------: |
| Mean      |     160.44 |
| Median    |        148 |
| P95       |        270 |
| P99       |       ~407 |
| Maximum   |      1,802 |

The analysis showed that retrieved documents can already provide substantial context. Therefore, aggressive document chunking was not introduced as a default preprocessing step, since splitting these documents could potentially remove useful semantic context and affect retrieval quality.

### Key Findings

The EDA established several important characteristics of the task:

1. The dataset is a **many-to-many query-document retrieval problem**.
2. Each query may have multiple relevant documents.
3. Documents can be relatively long, which is an important consideration for downstream RAG context construction.
4. The relevance judgments provide an offline ground truth for evaluating retrieval performance.

---

## 🎯 Sentence Embeddings

Sentence embeddings were used to represent queries and documents as dense numerical vectors in a shared semantic space.

The objective was to identify an embedding model that could effectively capture the semantic relationship between queries and relevant documents and provide a strong foundation for dense retrieval.

### Model Benchmarking

Several embedding models and retrieval configurations were evaluated using the same queries, documents, and relevance judgments.

The evaluation used:

* **Recall@10**
* **MRR@10**
* **nDCG@10**

| Model / Method                  |  Recall@10 |     MRR@10 |    nDCG@10 |
| ------------------------------- | ---------: | ---------: | ---------: |
| **Fine-tuned MiniLM + MNRL** 🏆 | **0.2288** | **0.3474** | **0.2103** |
| MiniLM pretrained baseline      |     0.2207 |     0.3431 |     0.2040 |
| MPNet (`all-mpnet-base-v2`)     |     0.0402 | **0.3612** |     0.1440 |
| BGE-base                        |     0.0384 |     0.3494 |     0.1386 |
| Nomic Embed v2                  |     0.0308 |     0.3002 |     0.1131 |
| MiniLM + Chunking (128/32)      |     0.0345 |     0.3320 |     0.1260 |

### Model Selection

The benchmark revealed an important trade-off between the retrieval metrics.

MPNet achieved the highest **MRR@10**, indicating strong performance in placing relevant documents at very high ranks. However, its **Recall@10** and **nDCG@10** were substantially lower than the MiniLM-based approaches.

The pretrained MiniLM baseline provided the strongest overall retrieval performance among the pretrained configurations. Fine-tuning MiniLM with **Multiple Negatives Ranking Loss (MNRL)** further improved all three metrics.

Therefore, the fine-tuned MiniLM model was selected as the final embedding model for the retrieval pipeline.

### Embedding Fine-Tuning

The selected MiniLM model was fine-tuned specifically for the query-to-document retrieval task.

The training data consisted of query-document pairs constructed from the relevance judgments, together with negative and hard-negative documents.

The final training setup used:

* **3,944 positive query-document pairs**
* **38,449 hard negatives**
* **384-dimensional embeddings**
* **Multiple Negatives Ranking Loss (MNRL)**

The fine-tuned model achieved:

| Metric    | Pretrained MiniLM | Fine-tuned MiniLM |
| --------- | ----------------: | ----------------: |
| Recall@10 |            0.2207 |        **0.2288** |
| MRR@10    |            0.3431 |        **0.3474** |
| nDCG@10   |            0.2040 |        **0.2103** |

This improvement established the fine-tuned MiniLM model as the final embedding model used in the subsequent dense retrieval and vector search stages.


### Dense Retrieval

The fine-tuned sentence embedding model was used to perform dense retrieval over the document corpus.

For each query, the query embedding was compared against the document embeddings, and the most semantically similar documents were retrieved based on vector similarity.

The retrieval pipeline can be summarized as:

```text
Query
  ↓
Query Embedding
  ↓
Similarity Search
  ↓
Top-K Documents
```

### Retrieval Results

The fine-tuned embedding model achieved the following results on the retrieval evaluation:

| Metric |     @10 |
| ------ | ------: |
| Recall |  0.2288 |
| MRR    |  0.3474 |
| nDCG   | 0.2103 |

The results demonstrate that task-specific fine-tuning improved the semantic retrieval capability of the embedding model compared with the initial pretrained baseline.

The retrieved Top-10 documents were subsequently used as candidates for the vector search and downstream RAG pipeline.

### Retrieval Evaluation

The retrieval system was evaluated using relevance judgments (qrels) to analyze not only overall retrieval performance, but also how relevant documents were distributed across the Top-10 results.

### Overall Performance

Using the fine-tuned MiniLM embedding model, the retrieval system achieved:

| Metric    |        @10 |
| --------- | ---------: |
| Recall@10 | **0.2288** |
| MRR@10    | **0.3474** |
| nDCG@10   | **0.2103** |

### Rank-wise Relevance

The relevance rate of retrieved documents decreased as the rank increased:

| Rank | Relevant Rate |
| ---: | ------------: |
|    1 |         22.1% |
|    2 |         17.4% |
|    3 |         13.4% |
|    4 |         12.1% |
|    5 |          8.5% |
|    6 |          9.2% |
|    7 |          7.2% |
|    8 |          7.4% |
|    9 |          5.6% |
|   10 |          5.7% |

The highest concentration of relevant documents occurs in the first few ranks, confirming that the retrieval model generally places more relevant candidates near the top of the ranking.

### Top-10 Analysis

Across the 1,000 test queries:

* **62.5%** of queries had at least one relevant document in the Top-10.
* The average number of relevant documents retrieved per query was **1.086**.
* **375 queries** retrieved no relevant document in the Top-10.
* **324 queries** retrieved exactly one relevant document.
* **181 queries** retrieved two relevant documents.
* **88 queries** retrieved three relevant documents.
* **24 queries** retrieved four relevant documents.
* **8 queries** retrieved five relevant documents.

This analysis showed that increasing the retrieval depth provides additional relevant candidates, but with diminishing returns as the rank increases.

The Top-10 retrieval results were therefore retained as the candidate set for the subsequent vector search and RAG experiments.

---

## 🤖 Vector Search

To make the retrieval stage more suitable for large-scale search, the document embeddings were indexed using vector search methods.

Two approaches were evaluated:

* **Exact Search**
* **HNSW (Hierarchical Navigable Small World)**

The goal was to compare exact nearest-neighbor search with approximate nearest-neighbor search in terms of retrieval quality and search efficiency.

### Exact Search vs HNSW

Both methods were evaluated using the same query embeddings, document embeddings, and relevance judgments.

| Method       |  Recall@10 |     MRR@10 |    nDCG@10 | Mean Search Time |
| ------------ | ---------: | ---------: | ---------: | ---------------: |
| Exact Search |     0.2207 |     0.3431 |     0.2040 |          1535 ms |
| **HNSW**     | **0.2187** | **0.3421** | **0.2031** |       **615 ms** |

HNSW achieved retrieval performance very close to exact search while reducing the mean search time from approximately **1535 ms to 615 ms**.

The Top-10 results also showed complete overlap in the evaluated comparison, indicating that HNSW reproduced the same retrieved candidates for the tested queries while providing substantially faster search.

### Selection

HNSW was selected for the subsequent retrieval pipeline because it provided a significant reduction in search time with only a negligible change in retrieval metrics.

This makes approximate nearest-neighbor search a more practical choice for scaling the semantic retrieval system while preserving retrieval quality.


---

## 📉Reranking

A cross-encoder reranker was evaluated to determine whether reordering the Top-10 candidates retrieved by the embedding model could improve retrieval quality.

The reranking pipeline was:

```text
Query
  ↓
Vector Search
  ↓
Top-10 Candidates
  ↓
Cross-Encoder Reranker
  ↓
Re-ranked Top-10
```

The evaluated reranker was:

```text
BAAI/bge-reranker-base
```

The reranker scores each query-document pair jointly, allowing it to model the interaction between the query and retrieved document more directly than the bi-encoder used during initial retrieval.

### Results

| Method                    |     MRR@10 |    nDCG@10 |
| ------------------------- | ---------: | ---------: |
| Vector Search             | **0.3431** | **0.2040** |
| Vector Search + Reranking |     0.3212 |     0.1264 |

The reranker decreased both MRR@10 and nDCG@10 compared with the original vector-search ranking.

This experiment showed that adding a reranking stage did not improve retrieval quality for this dataset. Therefore, the original vector-search ranking was retained for the downstream RAG pipeline.

The experiment also highlights an important property of reranking: a reranker can only reorder the retrieved candidates and cannot recover documents that were missed during the initial retrieval stage.

Vector search vs reranker

Reranking with the current model substantially reduced retrieval metrics.
<img width="768" height="432" alt="Vector search vs reranker" src="https://github.com/user-attachments/assets/440b5e30-b2c8-40b6-9a9a-e4430c8cca16" />


Therefore, reranking was not included in the final RAG pipeline.

---

## 🔧 RAG

The final stage of the system extends the semantic retrieval pipeline into a Retrieval-Augmented Generation (RAG) workflow.

The RAG pipeline combines retrieved documents with an instruction prompt and an instruction-tuned language model to generate answers grounded in the retrieved context.

```text
Query
  ↓
Vector Search
  ↓
Top-K Documents
  ↓
Context Construction
  ↓
Prompt Construction
  ↓
LLM Generation
  ↓
Answer
  ↓
RAG Evaluation
```

### Context Construction

The Top-10 documents retrieved by the vector search stage were analyzed to determine an appropriate context size for the generation model.

Different values of `K` were evaluated based on the number of relevant documents retrieved and the resulting context length.

|  K | Avg. Relevant Docs | Query Hit Rate | Avg. Words | P95 Words |
| -: | -----------------: | -------------: | ---------: | --------: |
|  1 |              0.221 |          0.221 |        153 |       254 |
|  3 |              0.529 |          0.412 |        470 |       681 |
|  5 |              0.735 |          0.504 |        794 |     1,110 |
| 10 |              1.086 |          0.625 |      1,604 |     2,220 |

Based on the trade-off between evidence coverage and context size, **K=5** was selected as the initial context size for the RAG pipeline.

The retrieved documents were concatenated into a structured context, with each document assigned a rank identifier.

### Prompt Construction

The retrieved context was combined with the original query using a controlled prompt template.

The prompt instructs the language model to:

* use only the provided context,
* answer the user's question based on the retrieved evidence,
* and abstain when the context does not contain sufficient information.

This design aims to reduce unsupported generation and make the source of the answer explicit.

### Generation

The generation stage uses:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model receives the constructed prompt containing both the retrieved context and the original query.

Generation is performed without additional fine-tuning of the language model. The model is used as an instruction-tuned generator on top of the existing retrieval pipeline.

### RAG Evaluation

The RAG evaluation is designed to evaluate both the retrieved evidence and the generated answers.

The evaluation is divided into several dimensions:

| Component          | Evaluation Focus                                                   |
| ------------------ | ------------------------------------------------------------------ |
| Context            | Whether the retrieved context contains relevant evidence           |
| Faithfulness       | Whether generated claims are supported by the retrieved context    |
| Answer Relevance   | Whether the generated answer addresses the query                   |
| Answer Correctness | Whether the answer is factually correct against a reference answer |

Retrieval quality has already been evaluated using the available relevance judgments. The remaining RAG evaluation focuses on the quality of the constructed context and generated answers.

Because the current retrieval dataset provides query-document relevance judgments rather than reference answers, classical answer-correctness evaluation
