# src/evaluate.py

import numpy as np


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(retrieved_ids, query_ids, qrels, k=10):

    recalls = []

    for i, query_id in enumerate(query_ids):

        retrieved = set(retrieved_ids[i, :k])

        relevant = set(qrels[query_id])

        if len(relevant) == 0:
            continue

        hits = len(retrieved & relevant)

        recalls.append(hits / len(relevant))

    return np.mean(recalls)


# ============================================================
# MRR@K
# ============================================================

def mrr_at_k(retrieved_ids, query_ids, qrels, k=10):

    reciprocal_ranks = []

    for i, query_id in enumerate(query_ids):

        retrieved = retrieved_ids[i, :k]

        relevant = set(qrels[query_id])

        rr = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):

            if doc_id in relevant:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks)


# ============================================================
# nDCG@K
# ============================================================

def ndcg_at_k(retrieved_ids, query_ids, qrels, k=10):

    ndcgs = []

    for i, query_id in enumerate(query_ids):

        retrieved = retrieved_ids[i, :k]

        relevant = set(qrels[query_id])

        # ----------------------------
        # DCG
        # ----------------------------

        dcg = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):

            if doc_id in relevant:
                dcg += 1.0 / np.log2(rank + 1)

        # ----------------------------
        # Ideal DCG
        # ----------------------------

        ideal_relevant = min(len(relevant), k)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, ideal_relevant + 1)
        )

        if idcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        ndcgs.append(ndcg)

    return np.mean(ndcgs)


# ============================================================
# Complete Evaluation
# ============================================================

def evaluate_retrieval(
    retrieved_ids,
    query_ids,
    qrels,
    k=10,
):

    recall = recall_at_k(
        retrieved_ids,
        query_ids,
        qrels,
        k,
    )

    mrr = mrr_at_k(
        retrieved_ids,
        query_ids,
        qrels,
        k,
    )

    ndcg = ndcg_at_k(
        retrieved_ids,
        query_ids,
        qrels,
        k,
    )

    return {
        f"Recall@{k}": recall,
        f"MRR@{k}": mrr,
        f"nDCG@{k}": ndcg,
    }
