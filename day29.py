# day29.py — LangChain Agents with ReAct pattern

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
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

agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of policy POL-002?"}]
})

# Print all messages to see full reasoning
for message in result["messages"]:
    print(f"{type(message).__name__}: {message.content}")
    print("---")
