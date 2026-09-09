# day46.py — Chroma vector database with persistence and metadata

# import warnings
# warnings.filterwarnings("ignore")

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

# ── Documents with metadata ───────────────────────────────────────────────
documents = [
    {"text": "A deductible is the amount you pay before insurance kicks in.", "category": "general", "type": "definition"},
    {"text": "A premium is the monthly or annual cost of your insurance policy.", "category": "general", "type": "definition"},
    {"text": "A copay is a fixed amount you pay for a specific healthcare service.", "category": "health", "type": "definition"},
    {"text": "Coinsurance is the percentage you pay after meeting your deductible.", "category": "health", "type": "definition"},
    {"text": "Collision coverage pays for damage to your car from an accident.", "category": "auto", "type": "coverage"},
    {"text": "Comprehensive coverage pays for non-collision damage like theft or weather.", "category": "auto", "type": "coverage"},
    {"text": "Liability coverage pays for damage you cause to others.", "category": "auto", "type": "coverage"},
    {"text": "A claim is a request to your insurer to pay for a covered loss.", "category": "general", "type": "process"},
]

# ── Set up Chroma ─────────────────────────────────────────────────────────
print("Starting...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
print("Chroma client created.")

try:
    chroma_client.delete_collection("insurance")
    print("Old collection deleted.")
except:
    print("No existing collection to delete.")

collection = chroma_client.create_collection("insurance")
print("Collection created.")

# ── Add documents ─────────────────────────────────────────────────────────
print("Starting embedding loop...")
for i, doc in enumerate(documents):
    print(f"  Getting embedding {i+1}/{len(documents)}...")
    embedding = get_embedding(doc["text"])
    print(f"  Got embedding, adding to Chroma...")
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[doc["text"]],
        metadatas=[{"category": doc["category"], "type": doc["type"]}]
    )
    print(f"  Added {i+1}/{len(documents)}")

print(f"\nTotal documents in collection: {collection.count()}")

# ── Search without filter ─────────────────────────────────────────────────
print("\n=== Search: 'car accident coverage' ===")
query_embedding = get_embedding("car accident coverage")
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}] {doc}")

# ── Search with metadata filter ───────────────────────────────────────────
print("\n=== Search: 'what do I pay' — filtered to AUTO only ===")
query_embedding2 = get_embedding("what do I pay")
results2 = collection.query(
    query_embeddings=[query_embedding2],
    n_results=3,
    where={"category": "auto"}
)
for doc, meta in zip(results2["documents"][0], results2["metadatas"][0]):
    print(f"  [{meta['category']}] {doc}")
