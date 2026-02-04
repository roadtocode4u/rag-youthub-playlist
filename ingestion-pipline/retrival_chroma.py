import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)  


# ----------------------------
# 1) USER QUESTION → EMBEDDING
# ----------------------------
def embed_query(client: OpenAI, question: str, model: str = "text-embedding-3-small"):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    resp = client.embeddings.create(
        model=model,
        input=question
    )
    return resp.data[0].embedding


# ----------------------------
# 2) SIMILARITY SEARCH IN CHROMA
# ----------------------------
def search_chroma(
    chroma_dir: str,
    collection_name: str,
    query_embedding,
    top_k: int = 5
):
    client_chroma = chromadb.PersistentClient(path=chroma_dir)
    collection = client_chroma.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results


# ----------------------------
# 3) RELEVANT CHUNKS → LLM ANSWER
# ----------------------------
def build_context(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # Smaller distance usually means more similar (depends on metric/index)
    lines = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        lines.append(
            f"[{i}] (source={source}, chunk={chunk_index}, distance={dist:.4f})\n{doc}"
        )

    return "\n\n".join(lines)


def ask_llm(client: OpenAI, question: str, context: str, model: str = "gpt-4o-mini") -> str:
    system_msg = (
        "You are a helpful assistant. Answer ONLY using the given context. "
        "If the context does not contain the answer, say: "
        "'I don't have enough information in the provided documents.'"
    )

    user_msg = f"""
CONTEXT (retrieved from documents):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Use only the context.
- Keep the answer short and clear.
- If unsure, say you don't have enough information.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


# ----------------------------
# MAIN
# ----------------------------
def main():
    # Make sure OPENAI_API_KEY is set in your terminal
    # export OPENAI_API_KEY="sk-..."
    client_openai = OpenAI()

    chroma_dir = "chroma_store"
    collection_name = "my_knowledge_base"

    question = input("Ask a question: ").strip()

    # 1) question → embedding
    q_embedding = embed_query(client_openai, question)

    # 2) similarity search → relevant chunks
    results = search_chroma(chroma_dir, collection_name, q_embedding, top_k=5)

    # 3) build context string
    context = build_context(results)

    print("\n--- Retrieved Chunks (Top Matches) ---\n")
    print(context)

    # 4) context + question → LLM answer
    answer = ask_llm(client_openai, question, context)

    print("\n--- Final Answer (LLM) ---\n")
    print(answer)


if __name__ == "__main__":
    main()
