# day30.py — Built-in tools with custom tool fallback

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

@tool
def get_customer_info(customer_id: str) -> str:
    """Get information about a customer by their ID."""
    customers = {
        "CUST-001": "John Doe, age 35, policies: POL-001, POL-002",
        "CUST-002": "Jane Smith, age 28, policies: POL-003",
    }
    return customers.get(customer_id, f"Customer {customer_id} not found")

tools = [get_policy_info, get_customer_info]

agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What policies does customer CUST-001 have and what is the status of each?"}]
})

for message in result["messages"]:
    print(f"{type(message).__name__}: {message.content}")
    print("---")
