from langchain_text_splitters import CharacterTextSplitter

# Sample HR Policy Document
hr_policy = """
Employees are entitled to 24 paid leaves per year. Unused leaves cannot be carried forward to the next financial year. Employees must apply for leave at least 3 days in advance through the HR portal.

The notice period is 60 days for confirmed employees. During the probation period, it is reduced to 30 days. Employees are expected to complete all handover documentation before their last working day.

Work from home is allowed up to 2 days per week. Employees must inform their manager 24 hours in advance. Remote work policy is subject to project requirements and manager approval.

Performance reviews are conducted twice a year - in June and December. Employees are rated on a scale of 1 to 5. A rating of 3 or above is required for annual increment eligibility.
"""

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(hr_policy)
print(f"\nTotal characters: {len(hr_policy)}")

# ============================================================
# Example 1: Basic Character Splitting
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 1: Basic Character Splitting")
print("chunk_size=200")
print("=" * 60)

splitter_basic = CharacterTextSplitter(
    chunk_size=200,
    separator="\n\n"  # Try to split at paragraph boundaries first
)

chunks_basic = splitter_basic.split_text(hr_policy)

for i, chunk in enumerate(chunks_basic, 1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

# ============================================================
# Example 2: Smaller chunks (more granular)
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 2: Smaller Chunks")
print("chunk_size=100")
print("=" * 60)

splitter_small = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "  # Split at spaces for smaller chunks
)

chunks_small = splitter_small.split_text(hr_policy)

for i, chunk in enumerate(chunks_small, 1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

# ============================================================
# Example 3: Using with Documents (for RAG pipelines)
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 3: Working with Document Objects")
print("=" * 60)

from langchain_core.documents import Document

# Create a document with metadata
doc = Document(
    page_content=hr_policy,
    metadata={"source": "hr_policy.pdf", "page": 1}
)

splitter_docs = CharacterTextSplitter(
    chunk_size=250,
    separator="\n\n"
)

# split_documents preserves metadata!
doc_chunks = splitter_docs.split_documents([doc])

for i, chunk in enumerate(doc_chunks, 1):
    print(f"\n--- Document Chunk {i} ---")
    print(f"Content: {chunk.page_content[:80]}...")
    print(f"Metadata: {chunk.metadata}")