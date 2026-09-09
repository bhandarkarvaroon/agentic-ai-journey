# day54.py — RAG with conversation memory

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── Load Chroma store ─────────────────────────────────────────────────────
vectorstore = Chroma(
    persist_directory="./chroma_insurance",
    embedding_function=embeddings,
    collection_name="insurance_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ── Step 1: Rephrase question using chat history ──────────────────────────
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and the latest user question,
rephrase the question into a standalone search query.
Return ONLY the rephrased question — no answers, no bullet points, no explanations.

Example:
Chat history: User asked about deductibles and premiums
Question: Can you summarise all three?
Rephrased: Summarise deductible, premium, and coinsurance in insurance."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Rephrase this question as a standalone search query: {question}")
])

rephrase_chain = rephrase_prompt | llm | StrOutputParser()

# ── Step 2: Answer using retrieved context ────────────────────────────────
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an insurance assistant. Answer using only the context.
If not in context, say 'I don't have that information.'

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

answer_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ── Conversational RAG function ───────────────────────────────────────────
def chat(question: str, chat_history: list) -> str:
    if chat_history:
        standalone_question = rephrase_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        print(f"  [Rephrased: {standalone_question}]")
    else:
        standalone_question = question

    answer = answer_chain.invoke({
        "question": standalone_question,
        "chat_history": chat_history
    })

    return answer

# ── Test multi-turn conversation ──────────────────────────────────────────
chat_history = []

questions = [
    "What is a deductible?",
    "How is it different from a premium?",
    "What about coinsurance?",
    "Can you summarise all three?"
]

for q in questions:
    print(f"You: {q}")
    answer = chat(q, chat_history)
    print(f"Assistant: {answer}\n")
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=answer))
