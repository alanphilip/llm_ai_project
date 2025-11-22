import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# 2. Our "Database" of documents
# In a real app, these would be rows in a database or chunks of PDFs.
documents = [
    "The best way to cook a steak is to sear it on high heat.",
    "Machine learning models require large datasets for training.",
    "The Golden State Warriors won the NBA championship.",
    "Generative AI can create images and text from prompts.",
    "To fix a flat tire, you need a jack and a lug wrench."
]

# 3. Helper function for embeddings (reused)
def get_embedding(text):
    return genai.embed_content(model="models/text-embedding-004",content=text)['embedding']

# 4. "Indexing" Step
# We pre-calculate embeddings for our database so we don't have to do it every time we search.
print("Indexing documents...")
doc_embeddings = []
for doc in documents:
    doc_embeddings.append({
        "text": doc,
        "vector": np.array(get_embedding(doc))
    })
print("Indexing complete!\n")

# 5. The Search Function
def search_engine(query):
    query_vector = np.array(get_embedding(query))

    results = []

    # Compare query against every document
    for doc_data in doc_embeddings:
        doc_vector = doc_data["vector"]

        # Calculate Cosine Similarity manually
        dot_product = np.dot(query_vector, doc_vector)
        norm_a = np.linalg.norm(query_vector)
        norm_b = np.linalg.norm(doc_vector)
        similarity = dot_product / (norm_a * norm_b)

        results.append((similarity, doc_data["text"]))

    # Sort results by score (highest first)
    results.sort(key=lambda x: x[0], reverse=True)

    # Print the Top Match
    print(f"Query: '{query}'")
    print(f"Top Match: '{results[0][1]}'")
    print(f"Confidence Score: {results[0][0]:.4f}")
    print("-" * 30)

# 6. Test it out!
# Notice that we don't use exact keywords found in the documents.
search_engine("How do I prepare beef?")
search_engine("Tell me about artificial intelligence.")
search_engine("sports news")