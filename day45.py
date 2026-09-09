# day45.py — FAISS vector store

import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import faiss

load_dotenv()

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ── Same documents as Day 44 ──────────────────────────────────────────────
documents = [
    "A deductible is the amount you pay before insurance kicks in.",
    "A premium is the monthly or annual cost of your insurance policy.",
    "A copay is a fixed amount you pay for a specific healthcare service.",
    "Coinsurance is the percentage you pay after meeting your deductible.",
    "An out-of-pocket maximum is the most you pay in a year before insurance covers 100%.",
    "A beneficiary is the person who receives insurance benefits.",
    "Collision coverage pays for damage to your car from an accident.",
    "Comprehensive coverage pays for non-collision damage like theft or weather.",
    "Liability coverage pays for damage you cause to others.",
    "A claim is a request to your insurer to pay for a covered loss.",
]

# ── Embed documents ───────────────────────────────────────────────────────
print("Embedding documents...")
embeddings = []
for i, doc in enumerate(documents):
    print(f"  Embedding {i+1}/{len(documents)}...")
    embeddings.append(get_embedding(doc))

print("All embeddings done.")

# Convert to numpy array — FAISS requires float32
embedding_matrix = np.array(embeddings, dtype=np.float32)
dimension = embedding_matrix.shape[1]
print(f"Matrix shape: {embedding_matrix.shape}")

# ── Build FAISS index ─────────────────────────────────────────────────────
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(embedding_matrix)
print(f"FAISS index built. Total vectors: {index.ntotal}")

# ── Search function ───────────────────────────────────────────────────────
def search(query: str, top_k: int = 3) -> None:
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)

    print(f"\nQuery: {query}")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  {i+1}. [{dist:.4f}] {documents[idx]}")

# ── Test ──────────────────────────────────────────────────────────────────
search("What do I pay monthly for insurance?")
search("What happens if my car gets stolen?")
search("How much do I pay when I visit a doctor?")
