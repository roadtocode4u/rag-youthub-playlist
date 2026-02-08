from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings  
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  

# HR Policy document with distinct topics for semantic grouping
hr_policy_text = """Leave Policy
Employees are entitled to 24 paid leaves per year.
Unused leaves cannot be carried forward to the next year.
Sick leave requires a medical certificate if taken for more than 2 days.
Maternity leave is 26 weeks and paternity leave is 2 weeks.

Notice Period and Resignation
The notice period is 60 days for confirmed employees.
During probation, the notice period is reduced to 30 days.
Employees must complete all handover documentation before leaving.
Exit interviews are mandatory for all departing employees.

Remote Work Guidelines
Work from home is allowed up to 3 days per week.
Employees must have a stable internet connection for remote work.
Core working hours are 10 AM to 4 PM regardless of location.
Manager approval is required for any remote work arrangement.

Health Insurance Benefits
The company provides comprehensive health insurance for all employees.
Coverage includes the employee, spouse, and up to 2 dependent children.
Dental and vision coverage can be added for an additional premium.
Pre-existing conditions are covered after a 90-day waiting period."""

# Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=90  # Higher = fewer, larger chunks (splits at biggest meaning jumps)
)

chunks = semantic_splitter.split_text(hr_policy_text)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
