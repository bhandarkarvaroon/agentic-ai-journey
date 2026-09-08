# day34.py — Agent with 3 custom tools

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic."""
    results = {
        "LangGraph": """LangGraph is a library for building stateful, multi-actor applications with LLMs.
        Key features: 1) Built on LangChain 2) Uses graph-based architecture with nodes and edges
        3) Supports cycles and branching 4) Great for complex agentic workflows
        5) Supports human-in-the-loop 6) Has built-in persistence and checkpointing.""",
        "RAG": "RAG stands for Retrieval Augmented Generation. It combines retrieval of external documents with LLM generation.",
        "LangChain": "LangChain is a framework for building LLM applications. It provides chains, agents, memory and tools."
    }
    for key in results:
        if key.lower() in query.lower():
            return results[key]
    return f"No results found for: {query}"

@tool
def summarise_text(text: str) -> str:
    """Summarise the given text into 3 bullet points."""
    response = llm.invoke([
        {"role": "system", "content": "Summarise the following text into 3 bullet points."},
        {"role": "user", "content": text}
    ])
    return response.content

@tool
def save_note(note: str) -> str:
    """Save a note to a file."""
    with open("notes.txt", "a") as f:
        f.write(note + "\n")
    return "Note saved successfully"

tools = [web_search, summarise_text, save_note]

agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research RAG and LangChain and save both summaries."}]
})
