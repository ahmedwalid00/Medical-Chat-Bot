initial_prompt = "\n".join([
        "You are a expert Medical assistant. Please read the following document chunk and answer the question ",
        "based on the content provided. Ensure your answer is based on the information from this chunk only.",
        "If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'",
        "Question: {question}",
        "",
        "Document Chunk: {document_chunk}",
        "",
        "Answer:"
])

map_prompt = "\n".join(
    [
        "You are an expert Medical assistant tasked to generate a final answer from the following contexts. ",
        "Please focus on the most relevant information related to the question.",
        "If the answer is not contained in the intermediate contexts, say 'NO ANSWER IS AVAILABLE'",
        "Question: {question}",
        "",
        "Document Chunk: {document_chunk}",
        "",
        "Final answer:"
    ]
)

