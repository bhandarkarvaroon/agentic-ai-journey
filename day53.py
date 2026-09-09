# day53.py — RAG retrieval chain

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── Load existing Chroma store (no re-embedding needed) ───────────────────
vectorstore = Chroma(
    persist_directory="./chroma_insurance",
    embedding_function=embeddings,
    collection_name="insurance_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── RAG prompt ────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an insurance assistant. Answer the question using 
only the context provided. If the answer is not in the context, say 
'I don't have that information.'

Context:
{context}"""),
    ("user", "{question}")
])

# ── Helper to format retrieved docs ──────────────────────────────────────
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── RAG chain ─────────────────────────────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Test ──────────────────────────────────────────────────────────────────
questions = [
    "What is a deductible?",
    "What does comprehensive coverage pay for?",
    "What is the capital of France?",  # not in context
]

for q in questions:
    print(f"Q: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}\n")
