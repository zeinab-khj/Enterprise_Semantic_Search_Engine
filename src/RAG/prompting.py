def build_prompt(query, context):
    return f"""You are a question-answering assistant.

Use the provided context to answer the question.
Base your answer only on the information contained in the context.
If the context does not contain enough information to answer the question,
say that the information is not available in the provided context.

Context:
{context}

Question:
{query}

Answer:
"""


prompts = [
    build_prompt(query, context)
    for query, context in zip(query, context)
]

