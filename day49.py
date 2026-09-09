# day49.py — Comparing chunk sizes for semantic search

import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of chunk_size words with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ── Load document ─────────────────────────────────────────────────────────
with open("insurance_guide.txt", "r") as f:
    full_text = f.read()

# ── Test 3 chunk sizes ────────────────────────────────────────────────────
query = "what do I pay when I visit a doctor?"

for chunk_size, overlap in [(10, 2), (20, 4), (40, 8)]:
    chunks = chunk_text(full_text, chunk_size, overlap)

    # Store in Chroma
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = f"chunks_{chunk_size}"

    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(collection_name)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )

    # Search
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    print(f"\n=== Chunk size: {chunk_size} words, Overlap: {overlap} words ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Query: {query}")
    for doc in results["documents"][0]:
        print(f"  → {doc}")
