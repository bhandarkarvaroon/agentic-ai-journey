# day33.py — Agent with memory using checkpointer

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def get_policy_info(policy_id: str) -> str:
    """Get information about an insurance policy by its ID."""
    policies = {
        "POL-001": "Auto insurance, premium $1200/year, active",
        "POL-002": "Home insurance, premium $800/year, active",
        "POL-003": "Health insurance, premium $2400/year, expired"
    }
    return policies.get(policy_id, f"Policy {policy_id} not found")

tools = [get_policy_info]

# MemorySaver stores conversation in memory
checkpointer = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=checkpointer)

# thread_id = session ID — same ID = same conversation
config = {"configurable": {"thread_id": "session-001"}}

# Turn 1
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of policy POL-001?"}]
}, config=config)
print(f"Turn 1: {result1['messages'][-1].content}")

# Turn 2 — references previous turn
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "What was the premium for that policy?"}]
}, config=config)
print(f"Turn 2: {result2['messages'][-1].content}")

# Turn 3 — asks about conversation history
result3 = agent.invoke({
    "messages": [{"role": "user", "content": "What was my first question?"}]
}, config=config)
print(f"Turn 3: {result3['messages'][-1].content}")

# New session — should not remember previous conversation
config2 = {"configurable": {"thread_id": "session-002"}}

result4 = agent.invoke({
    "messages": [{"role": "user", "content": "What was my first question?"}]
}, config=config2)
print(f"New session: {result4['messages'][-1].content}")

agent2 = create_react_agent(
    llm, 
    tools, 
    checkpointer=MemorySaver(),
    prompt="You are a helpful insurance assistant. If you don't have information or context, say 'I don't know' — never guess."
)

config3 = {"configurable": {"thread_id": "session-003"}}

result5 = agent2.invoke({
    "messages": [{"role": "user", "content": "What was my first question?"}]
}, config=config3)
print(f"With system prompt: {result5['messages'][-1].content}")
