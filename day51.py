# day51.py — Text splitting strategies

import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# ── Load document ─────────────────────────────────────────────────────────
loader = TextLoader("insurance_guide.txt")
docs = loader.load()
print(f"Original document: {len(docs[0].page_content)} characters\n")

# ── Splitter 1: CharacterTextSplitter ─────────────────────────────────────
print("=== CharacterTextSplitter (split on newline) ===")
char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20
)
char_chunks = char_splitter.split_documents(docs)
print(f"Number of chunks: {len(char_chunks)}")
for i, chunk in enumerate(char_chunks[:3]):
    print(f"  Chunk {i+1} ({len(chunk.page_content)} chars): {chunk.page_content[:80]}...")

print()

# ── Splitter 2: RecursiveCharacterTextSplitter ────────────────────────────
print("=== RecursiveCharacterTextSplitter ===")
for chunk_size, chunk_overlap in [(100, 20), (200, 40), (500, 100)]:
    print(f"  Testing chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = recursive_splitter.split_documents(docs)
    print(f"  → {len(chunks)} chunks")
    print(f"    First chunk: {chunks[0].page_content[:80]}...")

print()

# ── Best split ────────────────────────────────────────────────────────────
print("=== Best split (chunk_size=200, chunk_overlap=40) ===")
best_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)
best_chunks = best_splitter.split_documents(docs)
print(f"Total chunks: {len(best_chunks)}")
for i, chunk in enumerate(best_chunks):
    print(f"  Chunk {i+1}: {chunk.page_content[:100]}...")
