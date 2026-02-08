from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True) 

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Sample text to chunk
sample_text = """Introduction to Machine Learning
Machine learning is a subset of artificial intelligence that enables computers to learn from data.
It has transformed industries from healthcare to finance.
The field continues to grow rapidly with new breakthroughs every year.

Types of Machine Learning
Supervised learning uses labeled data to train models for prediction tasks.
Unsupervised learning finds hidden patterns in data without labels.
Reinforcement learning trains agents through rewards and penalties.

Real-World Applications
Self-driving cars use deep learning for object detection and navigation.
Netflix recommends movies using collaborative filtering algorithms.
Banks detect fraud by analyzing transaction patterns in real-time."""

# Create the prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{sample_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

# Get AI response
print("🤖 Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

# Split the text at the markers
chunks = marked_text.split("<<<SPLIT>>>")

# Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:  # Only keep non-empty chunks
        clean_chunks.append(cleaned)

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
