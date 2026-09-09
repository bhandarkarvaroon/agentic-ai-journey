# day50.py — Document loaders in LangChain

import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Loader 1: TextLoader ──────────────────────────────────────────────────
print("=== TextLoader ===")
loader = TextLoader("insurance_guide.txt")
docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Type: {type(docs[0])}")
print(f"Content preview: {docs[0].page_content[:100]}")
print(f"Metadata: {docs[0].metadata}")

# ── Loader 2: Manual Document creation ───────────────────────────────────
print("\n=== Manual Document ===")
manual_doc = Document(
    page_content="This is a sample insurance policy document.",
    metadata={"source": "manual", "author": "Varun", "year": 2026}
)
print(f"Content: {manual_doc.page_content}")
print(f"Metadata: {manual_doc.metadata}")

# ── Loader 3: Load multiple files ─────────────────────────────────────────
print("\n=== Loading multiple files ===")
files = ["insurance_guide.txt", "engineers.txt"]
all_docs = []
for file in files:
    try:
        loader = TextLoader(file)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded {file}: {len(docs)} document(s)")
    except Exception as e:
        print(f"Failed to load {file}: {e}")

print(f"\nTotal documents loaded: {len(all_docs)}")
for doc in all_docs:
    print(f"  Source: {doc.metadata.get('source')} — {len(doc.page_content)} chars")
