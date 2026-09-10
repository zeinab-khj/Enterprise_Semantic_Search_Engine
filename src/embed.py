from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATH = "models/sentence-transformer/final"

OUTPUT_DIR = Path("data/embeddings")

BATCH_SIZE = 32


# ============================================================
# 2. Load fine-tuned model
# ============================================================

model = SentenceTransformer(MODEL_PATH)

print("Model loaded.")
print("Embedding dimension:", model.get_sentence_embedding_dimension())


# ============================================================
# 3. Load corpus and queries
# ============================================================

corpus = load_dataset(
    "BeIR/scidocs",
    "corpus"
    )
queries = load_dataset(
        "BeIR/scidocs",
        "queries"
    )


# ============================================================
# 4. Prepare texts
# ============================================================

corpus_df = corpus["corpus"].to_pandas()
queries_df = queries["queries"].to_pandas()


# ============================================================
# 5. Create embeddings
# ============================================================

print("\nEncoding corpus...")

corpus_embeddings =model.encode(
    corpus_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)

print("\nEncoding queries...")

query_embeddings = model.encode(
    queries_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# ============================================================
# 6. Save embeddings
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(
    OUTPUT_DIR / "corpus_embeddings.npy",
    corpus_embeddings,
)

np.save(
    OUTPUT_DIR / "query_embeddings.npy",
    query_embeddings,
)


# ============================================================
# 7. Save IDs
# ============================================================

corpus["_id"].to_csv(
    OUTPUT_DIR / "corpus_ids.csv",
    index=False,
)

queries["_id"].to_csv(
    OUTPUT_DIR / "query_ids.csv",
    index=False,
)


# ============================================================
# 8. Summary
# ============================================================

print("\nEmbedding completed.")

print("Corpus embeddings:", corpus_embeddings.shape)
print("Query embeddings:", query_embeddings.shape)

print(f"\nSaved to: {OUTPUT_DIR}")
