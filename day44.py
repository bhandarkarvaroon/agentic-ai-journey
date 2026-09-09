# day44.py — Semantic search over a text corpus

import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Insurance knowledge base ──────────────────────────────────────────────
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
    "Underwriting is the process insurers use to evaluate risk.",
    "A policy is a contract between you and the insurance company.",
    "Reinsurance is when insurers buy insurance from other insurers.",
    "An actuary calculates insurance risks and premiums using statistics.",
    "A grace period is extra time allowed to pay your premium after the due date.",
    "Subrogation is the insurer's right to recover costs from a third party.",
    "An endorsement is an amendment to an existing insurance policy.",
    "A rider is an add-on to a policy that provides additional coverage.",
    "Loss of use coverage pays for living expenses if your home is uninhabitable.",
    "Gap insurance covers the difference between a car's value and what you owe."
]

# ── Embed all documents ───────────────────────────────────────────────────
print("Embedding documents...")
embedded_docs = [(doc, get_embedding(doc)) for doc in documents]
print(f"Embedded {len(embedded_docs)} documents.\n")

# ── Search function ───────────────────────────────────────────────────────
def search(query: str, top_k: int = 3) -> list[tuple[float, str]]:
    query_embedding = get_embedding(query)
    scores = [
        (cosine_similarity(query_embedding, emb), doc)
        for doc, emb in embedded_docs
    ]
    scores.sort(reverse=True)
    return scores[:top_k]

# ── Test queries ──────────────────────────────────────────────────────────
queries = [
    "What do I pay monthly for insurance?",
    "What happens if my car gets stolen?",
    "How much do I pay when I visit a doctor?"
]

for q in queries:
    print(f"Query: {q}")
    results = search(q, top_k=3)
    for score, doc in results:
        print(f"  {score:.4f} — {doc}")
    print()
