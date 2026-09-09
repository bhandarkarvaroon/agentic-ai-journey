# day48.py — Semantic search CLI over a text file

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

# ── Load text file ────────────────────────────────────────────────────────
print("Loading insurance_guide.txt...")
with open("insurance_guide.txt", "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(lines)} lines.")

# ── Set up Chroma ─────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection("insurance_guide")
except:
    pass

collection = chroma_client.create_collection("insurance_guide")

# ── Embed and store ───────────────────────────────────────────────────────
print("Embedding and storing...")
for i, line in enumerate(lines):
    embedding = get_embedding(line)
    collection.add(
        ids=[f"line_{i}"],
        embeddings=[embedding],
        documents=[line]
    )
    print(f"  {i+1}/{len(lines)} done")

print(f"\nAll {collection.count()} lines stored in Chroma.\n")

# ── Search CLI ────────────────────────────────────────────────────────────
print("Semantic Search Ready. Type 'quit' to exit.\n")

while True:
    query = input("Search: ")
    if query.lower() == "quit":
        break

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nTop 3 results:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"  {i+1}. {doc}")
    print()
