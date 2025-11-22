import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# Define embedding model
EMBEDDING_MODEL = "models/text-embedding-004"

# Generate the embedding
text_embedding = genai.embed_content(
    model=EMBEDDING_MODEL,
    content="The quick brown fox jumps over the lazy dog.",
    task_type="retrieval_document", # Optional: Tells the model how you plan to use the embedding
    title="Embedding Example"       # Optional: Only used with task_type="retrieval_document"
)

# View the results
embedding_vector = text_embedding['embedding']

print(f"Vector Length: {len(embedding_vector)}") # Usually 768 dimensions
print(f"First 5 numbers: {embedding_vector[:5]}")