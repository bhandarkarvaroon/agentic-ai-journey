# day52.py — Full ingestion pipeline: load → split → embed → store in Chroma

import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# ── Step 1: Load ──────────────────────────────────────────────────────────
print("Step 1: Loading document...")
loader = TextLoader("insurance_guide.txt")
docs = loader.load()
print(f"  Loaded {len(docs)} document(s), {len(docs[0].page_content)} characters")

# ── Step 2: Split ─────────────────────────────────────────────────────────
print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(docs)
print(f"  Created {len(chunks)} chunks")

# ── Step 3: Embed + Store in Chroma ──────────────────────────────────────
print("Step 3: Embedding and storing in Chroma...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_insurance",
    collection_name="insurance_rag"
)
print(f"  Stored {vectorstore._collection.count()} chunks in Chroma")

# ── Step 4: Test retrieval ────────────────────────────────────────────────
print("\nStep 4: Testing retrieval...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

queries = [
    "What do I pay monthly for insurance?",
    "What happens if my car gets stolen?",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content[:100]}...")
