from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATH = "models/sentence-transformer/final"

CORPUS_FILE = "data/corpus.csv"
QUERIES_FILE = "data/queries.csv"

OUTPUT_DIR = Path("data/embeddings")

BATCH_SIZE = 64


# ============================================================
# 2. Load fine-tuned model
# ============================================================

model = SentenceTransformer(MODEL_PATH)

print("Model loaded.")
print("Embedding dimension:", model.get_sentence_embedding_dimension())


# ============================================================
# 3. Load corpus and queries
# ============================================================

corpus = pd.read_csv(CORPUS_FILE)
queries = pd.read_csv(QUERIES_FILE)

print(f"Corpus size: {len(corpus)}")
print(f"Query size: {len(queries)}")


# ============================================================
# 4. Prepare texts
# ============================================================

corpus_texts = (
    corpus["title"].fillna("") + " " +
    corpus["text"].fillna("")
).tolist()

query_texts = queries["text"].fillna("").tolist()


# ============================================================
# 5. Create embeddings
# ============================================================

print("\nEncoding corpus...")

corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("\nEncoding queries...")

query_embeddings = model.encode(
    query_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
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
