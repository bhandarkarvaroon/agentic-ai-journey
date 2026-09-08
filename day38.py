# day38.py — LangChain callbacks for logging and monitoring

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Custom callback handler ───────────────────────────────────────────────
class LoggingCallback(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM called")

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM done — "
                  f"input tokens: {usage.get('prompt_tokens', 'N/A')}, "
                  f"output tokens: {usage.get('completion_tokens', 'N/A')}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM done")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Tool called: {tool_name} with input: {input_str[:50]}")

    def on_tool_end(self, output, **kwargs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Tool returned: {str(output)[:80]}")

# ── Tools ─────────────────────────────────────────────────────────────────
@tool
def get_policy_info(policy_id: str) -> str:
    """Get information about an insurance policy by its ID."""
    policies = {
        "POL-001": "Auto insurance, premium $1200/year, active",
        "POL-002": "Home insurance, premium $800/year, active",
    }
    return policies.get(policy_id, f"Policy {policy_id} not found")

tools = [get_policy_info]

# ── Callback ──────────────────────────────────────────────────────────────
callback = LoggingCallback()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

# ── Test callback directly on LLM ────────────────────────────────────────
print("=== Testing callback directly ===")
test_response = llm.invoke([{"role": "user", "content": "Say hello"}])
print(f"Direct LLM response: {test_response.content[:50]}")
print()

# ── Agent ─────────────────────────────────────────────────────────────────
agent = create_react_agent(llm, tools)

print("=== Agent Run ===\n")
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of policy POL-001?"}]
})

print(f"\n=== Final Answer ===")
print(result["messages"][-1].content)
