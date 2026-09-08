# day41.py — Agent with memory + logging callback

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ── Logging callback ──────────────────────────────────────────────────────
class LoggingCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM called")

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM done — "
                  f"input: {usage.get('prompt_tokens', 'N/A')} tokens, "
                  f"output: {usage.get('completion_tokens', 'N/A')} tokens")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM done")

# ── Tools ─────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[LoggingCallback()])

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
    open("notes.txt", "w").close()  # clear file
    with open("notes.txt", "a") as f:
        f.write(note + "\n")
    return "Note saved successfully"

tools = [web_search, summarise_text, save_note]

# ── Agent with memory ─────────────────────────────────────────────────────
checkpointer = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "session-day41"}}

# ── Turn 1 ────────────────────────────────────────────────────────────────
print("=== Turn 1 ===")
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "Research LangGraph and save a summary."}]
}, config=config)
print(f"Answer: {result1['messages'][-1].content}\n")

# ── Turn 2 — references previous turn ────────────────────────────────────
print("=== Turn 2 ===")
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Now research RAG and save that summary too."}]
}, config=config)
print(f"Answer: {result2['messages'][-1].content}\n")

# ── Turn 3 — tests memory ────────────────────────────────────────────────
print("=== Turn 3 ===")
result3 = agent.invoke({
    "messages": [{"role": "user", "content": "What topics did I ask you to research?"}]
}, config=config)
print(f"Answer: {result3['messages'][-1].content}\n")

# ── Check notes.txt ───────────────────────────────────────────────────────
print("=== notes.txt ===")
with open("notes.txt", "r") as f:
    print(f.read())
