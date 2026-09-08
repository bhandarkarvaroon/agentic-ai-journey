# day36.py — Conversation memory types (modern LangChain approach)

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Shared conversation turns
conversation = [
    "What is a deductible?",
    "What is a premium?",
    "What is a copay?",
    "What is coinsurance?",
    "What is an out-of-pocket maximum?",
    "What is a beneficiary?",
]

# ── 1. Full Buffer — keeps everything ────────────────────────────────────
print("=== Full Buffer Memory ===")
checkpointer = MemorySaver()
agent = create_react_agent(llm, [], checkpointer=checkpointer)
config = {"configurable": {"thread_id": "buffer-session"}}

for question in conversation:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)

# Check how many messages are stored
state = agent.get_state(config)
print(f"Total messages in memory: {len(state.values['messages'])}")
print(f"Last answer: {state.values['messages'][-1].content[:100]}")

print()

# ── 2. Window — manually keep last 2 exchanges ───────────────────────────
print("=== Window Memory (last 2 exchanges) ===")
window = []
for question in conversation:
    window.append({"role": "user", "content": question})
    response = llm.invoke(window)
    window.append({"role": "assistant", "content": response.content})
    # Keep only system + last 4 messages (2 exchanges)
    if len(window) > 4:
        window = window[-4:]

print(f"Messages in window: {len(window)}")
print(f"Last answer: {window[-1]['content'][:100]}")

print()

# ── 3. Summary — compress old messages ───────────────────────────────────
print("=== Summary Memory ===")
messages = []
summary = ""

for question in conversation:
    if summary:
        context = [{"role": "system", "content": f"Previous conversation summary: {summary}"}]
    else:
        context = []
    
    context.append({"role": "user", "content": question})
    response = llm.invoke(context)
    messages.append((question, response.content))
    
    # Summarise every 2 exchanges
    if len(messages) % 2 == 0:
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in messages])
        summary_response = llm.invoke([
            {"role": "system", "content": "Summarise this conversation in 2 sentences."},
            {"role": "user", "content": history}
        ])
        summary = summary_response.content

print(f"Final summary: {summary[:200]}")
