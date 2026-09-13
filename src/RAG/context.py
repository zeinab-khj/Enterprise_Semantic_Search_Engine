K = 5

# Top-K documents for each query
topk_doc_indices = indices[:, :K]

def build_context(doc_indices, corpus_df):
    context_parts = []

    for rank, doc_idx in enumerate(doc_indices, start=1):
        doc_text = corpus_df.iloc[doc_idx]["text"]

        context_parts.append(
            f"[Document {rank}]\n{doc_text}"
        )

    return "\n\n".join(context_parts)


# Build context for all queries
contexts = [
    build_context(doc_indices, corpus_df)
    for doc_indices in topk_doc_indices
]

print("Number of contexts:", len(contexts))
