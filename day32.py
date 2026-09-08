# day32.py — OpenAI Functions agent vs ReAct agent

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

# ── Tools ─────────────────────────────────────────────────────────────────
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
def calculate_premium(age: int, coverage_amount: int) -> str:
    """Calculate insurance premium based on age and coverage amount."""
    base_rate = 0.02
    age_factor = 1 + (age - 25) * 0.01 if age > 25 else 1
    premium = coverage_amount * base_rate * age_factor
    return f"Estimated annual premium: ${premium:,.2f}"

tools = [get_policy_info, calculate_premium]

# ── Agent with gpt-4o-mini (uses OpenAI tool calling natively) ────────────
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools)

question = "What is the status of policy POL-001 and what would the premium be for a 40 year old with $150,000 coverage?"

result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

# Print all messages to see tool calling in action
print("=== Message flow ===")
for msg in result["messages"]:
    name = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"{name}: [calling tools: {[tc['name'] for tc in msg.tool_calls]}]")
    else:
        print(f"{name}: {msg.content[:100] if msg.content else '(empty)'}")
print()
print("=== Final Answer ===")
print(result["messages"][-1].content)
