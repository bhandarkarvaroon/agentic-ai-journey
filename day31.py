# day31.py — Custom tools with @tool decorator and StructuredTool

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Tool 1 — single parameter, use @tool decorator
@tool
def get_policy_coverage(policy_id: str) -> str:
    """Get coverage details for an insurance policy by its ID."""
    coverages = {
        "POL-001": "Liability: $100,000 | Collision: $50,000 | Comprehensive: $50,000",
        "POL-002": "Dwelling: $300,000 | Personal Property: $50,000 | Liability: $100,000",
        "POL-003": "Hospitalization: $500,000 | Outpatient: $50,000 | Prescription: $10,000"
    }
    return coverages.get(policy_id, f"No coverage found for {policy_id}")

# Tool 2 — multiple parameters, use StructuredTool
class PremiumInput(BaseModel):
    age: int = Field(description="Age of the policyholder")
    coverage_amount: int = Field(description="Coverage amount in dollars")

def calculate_premium(age: int, coverage_amount: int) -> str:
    """Calculate insurance premium based on age and coverage amount."""
    base_rate = 0.02
    age_factor = 1 + (age - 25) * 0.01 if age > 25 else 1
    premium = coverage_amount * base_rate * age_factor
    return f"Estimated annual premium: ${premium:,.2f}"

premium_tool = StructuredTool.from_function(
    func=calculate_premium,
    name="calculate_premium",
    description="Calculate insurance premium based on age and coverage amount",
    args_schema=PremiumInput
)

tools = [get_policy_coverage, premium_tool]

agent = create_react_agent(llm, tools)

# Test 1 — single tool
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What is the coverage for policy POL-002?"}]
})
print("Test 1:", result1["messages"][-1].content)

print("\n---\n")

# Test 2 — multi-parameter tool
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate the premium for a 35 year old with $200,000 coverage."}]
})
print("Test 2:", result2["messages"][-1].content)
