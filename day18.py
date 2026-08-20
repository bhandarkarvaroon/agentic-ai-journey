# day18.py — Function/tool calling with OpenAI SDK

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_policy_summary",
            "description": "Get the summary of an insurance policy by policy ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "The insurance policy ID"
                    }
                },
                "required": ["policy_id"]
            }
        }
    }
]

# Fake tool implementation
def get_policy_summary(policy_id: str) -> str:
    return f"Policy {policy_id}: Auto insurance, coverage $50,000, premium $1,200/year, active."

messages = [
    {"role": "system", "content": "You are an insurance assistant."},
    {"role": "user", "content": "Can you get me the summary for policy POL-123?"}
]

# First call — model decides to use a tool
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

# Check if model wants to call a tool
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_args = json.loads(tool_call.function.arguments)

print(f"Model wants to call: {function_name} with args: {function_args}")

# Execute the function
result = get_policy_summary(**function_args)
print(f"Tool result: {result}")

# Send result back to model
messages.append(response.choices[0].message)  # add assistant's tool call
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": result
})

# Get final response
final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

print(f"\nFinal answer: {final_response.choices[0].message.content}")
