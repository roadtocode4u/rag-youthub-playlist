import os
import re
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv(override=True)  

from pypdf import PdfReader
import chromadb
from openai import OpenAI


# ----------------------------
# 1) TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(path: str) -> str:
    print(f"  📄 Extracting PDF: {os.path.basename(path)}")
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    full_text = "\n".join(parts)
    print(f"     ✓ Extracted {len(full_text)} characters from {len(reader.pages)} pages")
    return full_text


def extract_text_from_txt(path: str) -> str:
    print(f"  📝 Extracting text file: {os.path.basename(path)}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"     ✓ Extracted {len(text)} characters")
    return text


def load_documents_from_folder(folder: str) -> List[Dict]:
    """
    Returns list of:
    { "id": str, "text": str, "metadata": dict }
    """
    print(f"\n📂 Loading documents from: {folder}")
    print("=" * 50)
    docs = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isdir(path):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            raw_text = extract_text_from_pdf(path)
        elif ext in [".txt", ".md"]:
            raw_text = extract_text_from_txt(path)
        else:
            continue

        docs.append(
            {
                "id": filename,
                "text": raw_text,
                "metadata": {"source": filename, "path": path},
            }
        )
    print(f"\n✅ Loaded {len(docs)} documents")
    return docs


# ----------------------------
# 2) CLEANING
# ----------------------------
def clean_text(text: str) -> str:
    original_len = len(text)
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    cleaned = text.strip()
    print(f"  🧹 Cleaned text: {original_len} → {len(cleaned)} characters")
    return cleaned


# ----------------------------
# 3) CHUNKING
# ----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Beginner-friendly char-based chunking.
    """
    print(f"  ✂️  Chunking text (size={chunk_size}, overlap={overlap})")
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    print(f"     ✓ Created {len(chunks)} chunks")
    return chunks


# ----------------------------
# 4) EMBEDDINGS (OpenAI)
# ----------------------------
def get_openai_embeddings(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    Calls OpenAI embeddings endpoint in one batch.
    Input can be a list of strings. :contentReference[oaicite:3]{index=3}
    """
    print(f"  🧠 Generating embeddings for {len(texts)} texts using {model}...")
    # OpenAI embeddings API doesn't accept empty strings
    safe_texts = [t if t.strip() else " " for t in texts]

    resp = client.embeddings.create(
        model=model,
        input=safe_texts,
    )

    # resp.data is aligned with inputs
    embeddings = [item.embedding for item in resp.data]
    print(f"     ✓ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")
    return embeddings


# ----------------------------
# 5) VECTOR DB (Chroma)
# ----------------------------
def ingest_to_chroma(
    docs: List[Dict],
    persist_dir: str = "chroma_store",
    collection_name: str = "my_knowledge_base",
    embedding_model: str = "text-embedding-3-small",
):
    # OpenAI client reads OPENAI_API_KEY from env automatically
    client_openai = OpenAI()

    # Chroma persistent client
    print(f"\n💾 Setting up Chroma vector database")
    print("=" * 50)
    client_chroma = chromadb.PersistentClient(path=persist_dir)
    collection = client_chroma.get_or_create_collection(name=collection_name)
    print(f"✓ Collection '{collection_name}' ready\n")

    all_ids = []
    all_texts = []
    all_metadatas = []

    for doc_idx, doc in enumerate(docs, 1):
        print(f"\n📄 Processing document {doc_idx}/{len(docs)}: {doc['id']}")
        print("-" * 50)
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}::chunk_{i}"
            metadata = {**doc["metadata"], "chunk_index": i}

            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append(metadata)

    if not all_texts:
        print("❌ No chunks created. Check your documents.")
        return

    # Batch: embeddings + insert into Chroma
    print(f"\n🔄 Inserting {len(all_texts)} chunks into Chroma in batches")
    print("=" * 50)
    batch_size = 64  # safe + beginner friendly
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(all_texts), batch_size), 1):
        batch_texts = all_texts[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_metas = all_metadatas[i : i + batch_size]

        print(f"\nBatch {batch_num}/{total_batches}: Processing {len(batch_texts)} chunks")
        embeddings = get_openai_embeddings(client_openai, batch_texts, embedding_model)

        print(f"  💾 Inserting batch into Chroma...")
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=embeddings,
        )
        print(f"     ✓ Batch {batch_num} inserted successfully")

    print(f"\n" + "=" * 50)
    print(f"✅ Ingested {len(all_texts)} chunks into '{collection_name}'")
    print(f"📁 Chroma stored at: {persist_dir}")
    print(f"🧠 Embeddings model: {embedding_model}")


def main():
    folder = "data"
    docs = load_documents_from_folder(folder)

    if not docs:
        print("❌ No supported docs found in ./data (use .pdf, .txt, .md)")
        return

    ingest_to_chroma(docs)


if __name__ == "__main__":
    main()