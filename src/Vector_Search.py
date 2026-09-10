import faiss


def build_hnsw_index(corpus_embeddings, M=32, ef_search=64):
    """
    Build an HNSW index for approximate nearest-neighbor search.
    """

    dimension = corpus_embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        M,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efSearch = ef_search

    index.add(corpus_embeddings)

    return index


def search(index, query_embeddings, top_k=10):
    """
    Retrieve top-k nearest documents for each query.
    """

    scores, indices = index.search(
        query_embeddings,
        top_k,
    )

    return scores, indices
