# day25.py - Chat models and message types

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# -- Part 1: Sending messages directly --------------

messages = [
    SystemMessage(content="You are an insurace assistant."),
    HumanMessage(content="What is a deductible?")
]

response = llm.invoke(messages)
print(type(response))
print(response.content)

# ── Part 2: Multi-turn conversation with message history ─────────────────
history = [
    SystemMessage(content="You are an insurance assistant. Be concise.")
]

# Turn 1
history.append(HumanMessage(content="What is a premium?"))
response1 = llm.invoke(history)
history.append(response1)  # response1 is already an AIMessage
print(f"Turn 1: {response1.content}\n")

# Turn 2
history.append(HumanMessage(content="How is it different from a deductible?"))
response2 = llm.invoke(history)
history.append(response2)
print(f"Turn 2: {response2.content}\n")

# Print full history
print("=== Full conversation history ===")
for msg in history:
    print(f"{type(msg).__name__}: {msg.content[:80]}...")
