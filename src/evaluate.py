# src/evaluate.py

import numpy as np


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    recall_scores = []
    for query_idx, labels in enumerate(fine_tuned_retrieval_labels):

        query_id = queries_df.iloc[query_idx]["_id"]
        total_relevant = len(qrels_dict[query_id])
    
        # -------------------------
        # Recall@10
        # -------------------------
        retrieved_relevant = sum(labels)
    
        recall = retrieved_relevant / total_relevant
        recall_scores.append(recall)
      return recall_scores


def mrr_at_k(retrieved_ids, relevant_ids, k=10):
    mrr_scores = []
    for query_idx, labels in enumerate(fine_tuned_retrieval_labels):

        query_id = queries_df.iloc[query_idx]["_id"]
        total_relevant = len(qrels_dict[query_id])
        for rank, label in enumerate(labels, start=1):
            if label == 1:
                reciprocal_rank = 1 / rank
                break
    
        mrr_scores.append(reciprocal_rank)   
      return mrr_scores


def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
     ndcg_scores = []
     for query_idx, labels in enumerate(fine_tuned_retrieval_labels):

        query_id = queries_df.iloc[query_idx]["_id"]
        total_relevant = len(qrels_dict[query_id])

         dcg = 0.0

          for rank, label in enumerate(labels, start=1):
              if label == 1:
                  dcg += 1 / math.log2(rank + 1)
      
          ideal_relevant = min(total_relevant, len(labels))
      
          idcg = 0.0
      
          for rank in range(1, ideal_relevant + 1):
              idcg += 1 / math.log2(rank + 1)
      
          ndcg = dcg / idcg if idcg > 0 else 0.0
          ndcg_scores.append(ndcg)
       return ndcg_scores



def evaluate_retrieval(
    retrieved_ids,
    qrels,
    k=10
):
    recall = recall_at_k(
        retrieved_ids,
        qrels,
        k
    )

    mrr = mrr_at_k(
        retrieved_ids,
        qrels,
        k
    )

    ndcg = ndcg_at_k(
        retrieved_ids,
        qrels,
        k
    )

    return {
        f"Recall@{k}": recall,
        f"MRR@{k}": mrr,
        f"nDCG@{k}": ndcg,
    }
