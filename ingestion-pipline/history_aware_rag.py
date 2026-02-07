from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)  

# ----------------------------
# 1) Connect to Chroma (your existing persisted DB from ingestion)
# ----------------------------
PERSIST_DIR = "chroma_store"  # <-- use your folder (same as ingestion)
COLLECTION_NAME = "my_knowledge_base"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

# ----------------------------
# 2) LLM (chat model)
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Store conversation history (messages)
chat_history = []  # [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]


# ----------------------------
# Helper: rewrite question using history (History-aware step)
# ----------------------------
def rewrite_question(user_question: str) -> str:
    """
    If history exists, rewrite follow-up questions into a standalone searchable question.
    If no history, return original question.
    """
    if not chat_history:
        return user_question

    rewrite_prompt = [
        SystemMessage(
            content=(
                "You are a question rewriter for conversational RAG.\n"
                "Rewrite the user's latest question into a standalone question that includes any missing context.\n"
                "Rules:\n"
                "- Do NOT answer.\n"
                "- Return ONLY the rewritten question.\n"
                "- If already standalone, return it as-is."
            )
        ),
        *chat_history[-8:],  # last 4 turns (user+assistant) to keep it cheap
        HumanMessage(content=f"Latest user question: {user_question}"),
    ]

    rewritten = llm.invoke(rewrite_prompt).content.strip()
    return rewritten


# ----------------------------
# Helper: retrieve top-k documents
# ----------------------------
def retrieve_docs(search_question: str, k: int = 3):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(search_question)
    return docs


# ----------------------------
# Helper: answer using retrieved docs only
# ----------------------------
def answer_from_docs(user_question: str, docs):
    context = "\n\n".join(
        [f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = [
        SystemMessage(
            content=(
                "You are a helpful assistant.\n"
                "Answer ONLY using the provided documents.\n"
                "If the answer is not found, say:\n"
                "'I don't have enough information to answer that based on the provided documents.'\n"
                "Keep the answer clear and short."
            )
        ),
        HumanMessage(
            content=(
                f"User question:\n{user_question}\n\n"
                f"Documents:\n{context}"
            )
        ),
    ]

    return llm.invoke(prompt).content.strip()


# ----------------------------
# Main: one question flow
# ----------------------------
def ask_question(user_question: str):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question using history (history-aware)
    search_question = rewrite_question(user_question)
    print(f"🔎 Searching for: {search_question}")

    # Step 2: Retrieve relevant docs
    docs = retrieve_docs(search_question, k=3)

    print(f"📄 Found {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  {i}. {preview}...")

    # Step 3: Generate answer using docs only
    answer = answer_from_docs(user_question, docs)

    print(f"\n✅ Answer:\n{answer}")

    # Step 4: Save to history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer


# ----------------------------
# Chat loop
# ----------------------------
def start_chat():
    print("✅ History-Aware RAG Chat (LangChain + Chroma). Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ["quit", "exit", "/bye"]:
            print("Goodbye!")
            break
        if not question:
            continue
        ask_question(question)


if __name__ == "__main__":
    start_chat()
