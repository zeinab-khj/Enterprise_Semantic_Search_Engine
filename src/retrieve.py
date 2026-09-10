import numpy as np


def retrieve(query_embeddings, corpus_embeddings, top_k=10):
    """
    Exact / brute-force retrieval.
    """

    # similarity between every query and every corpus vector
    scores = query_embeddings @ corpus_embeddings.T

    # top-k documents for each query
    top_k_indices = np.argsort(
        -scores,
        axis=1
    )[:, :top_k]

    top_k_scores = np.take_along_axis(
        scores,
        top_k_indices,
        axis=1
    )

    return top_k_scores, top_k_indices
