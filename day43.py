# day43.py — Embeddings: converting text to vectors

import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Convert text to an embedding vector."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate similarity between two vectors. 1 = identical, 0 = unrelated."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Generate embeddings
texts = [
    "insurance policy covers collision damage",
    "auto coverage for vehicle accidents",
    "monthly premium payment schedule",
    "cooking recipes for dinner",
    "football match highlights"
]

print("Generating embeddings...")
embeddings = [(text, get_embedding(text)) for text in texts]

# Check vector length
print(f"Embedding dimensions: {len(embeddings[0][1])}")

# Compare similarity
query = "car accident coverage"
print(f"\nQuery: '{query}'")
query_embedding = get_embedding(query)

print("\nSimilarity scores:")
for text, emb in embeddings:
    score = cosine_similarity(query_embedding, emb)
    print(f"  {score:.4f} — {text}")
