# day47.py — Cosine similarity vs dot product vs Euclidean distance

import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ── Documents ─────────────────────────────────────────────────────────────
documents = [
    "A premium is the monthly cost of your insurance policy.",
    "Auto coverage protects your vehicle from accidents.",
    "A deductible is what you pay before insurance kicks in.",
    "Football match highlights from last weekend.",
    "Cooking recipes for a healthy dinner.",
]

query = "How much do I pay for my car insurance?"

print(f"Query: {query}\n")
print("Embedding query and documents...")
query_emb = get_embedding(query)
doc_embeddings = [(doc, get_embedding(doc)) for doc in documents]

print("\n{:<55} {:>10} {:>12} {:>12}".format("Document", "Cosine", "Dot Product", "Euclidean"))
print("-" * 95)

for doc, emb in doc_embeddings:
    cos = cosine_similarity(query_emb, emb)
    dot = dot_product(query_emb, emb)
    euc = euclidean_distance(query_emb, emb)
    print("{:<55} {:>10.4f} {:>12.4f} {:>12.4f}".format(doc[:54], cos, dot, euc))

print("\nNote: Cosine & Dot — higher is better. Euclidean — lower is better.")
