import time
import numpy as np


def benchmark_exact(
    query_embeddings,
    corpus_embeddings,
    top_k=10,
    warmup=3,
    runs=5,
):
    # Convert tensors to CPU NumPy arrays if necessary
    if hasattr(query_embeddings, "detach"):
        query_embeddings = query_embeddings.detach().cpu().numpy()

    if hasattr(corpus_embeddings, "detach"):
        corpus_embeddings = corpus_embeddings.detach().cpu().numpy()

    # Warm-up
    for _ in range(warmup):
        scores = query_embeddings @ corpus_embeddings.T
        np.argsort(-scores, axis=1)[:, :top_k]

    # Benchmark
    times = []

    for _ in range(runs):
        start = time.perf_counter()

        scores = query_embeddings @ corpus_embeddings.T
        np.argsort(-scores, axis=1)[:, :top_k]

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_seconds": np.mean(times),
        "std_seconds": np.std(times),
        "mean_ms": np.mean(times) * 1000,
    }


def benchmark_hnsw(
    index,
    query_embeddings,
    top_k=10,
    warmup=3,
    runs=5,
):
    # FAISS CPU index expects NumPy arrays
    if hasattr(query_embeddings, "detach"):
        query_embeddings = query_embeddings.detach().cpu().numpy()

    # Warm-up
    for _ in range(warmup):
        index.search(query_embeddings, top_k)

    # Benchmark
    times = []

    for _ in range(runs):
        start = time.perf_counter()

        index.search(query_embeddings, top_k)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_seconds": np.mean(times),
        "std_seconds": np.std(times),
        "mean_ms": np.mean(times) * 1000,
    }


def benchmark_retrieval(
    query_embeddings,
    corpus_embeddings,
    hnsw_index,
    top_k=10,
):
    exact_results = benchmark_exact(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        top_k=top_k,
    )

    hnsw_results = benchmark_hnsw(
        index=hnsw_index,
        query_embeddings=query_embeddings,
        top_k=top_k,
    )

    return {
        "Exact": exact_results,
        "HNSW": hnsw_results,
    }


if __name__ == "__main__":
    print("Benchmark module loaded.")
    benchmark_results = benchmark_retrieval(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        hnsw_index=index,
        top_k=10,
    )
    
    print("Exact:")
    print(benchmark_results["Exact"])
    
    print("\nHNSW:")
    print(benchmark_results["HNSW"])
