# day19.py - System prompts and few-shot prompting

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAIAPI_KEY"))

def ask(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )
    return response.choices[0].message.content

# Same question, 3 different system prompts
question = "What is an API?"

print("=== Casual ===")
print(ask("You are a friendly assistant who explains things simply.", question))

print("\n=== Technical ===")
print(ask("You are a senior software architect. Be precise and technical.", question))

print("\n=== Insurance domain ===")
print(ask("You are an insurance technology expert. Frame everything in insurance context.", question))

# Few-shot — classify news headlines
print("\n=== Few-shot: Classify news headlines ===")

few_shot_system = """You classify news headlines into: POLITICS, SPORTS, FINANCE, TECHNOLOGY.
Reply with just the category, nothing else.

Examples:
User: Prime Minister announces new budget reforms
Assistant: POLITICS

User: Stock markets hit record high amid inflation fears
Assistant: FINANCE

User: Manchester United wins Champions League final
Assistant: SPORTS"""

headlines = [
    "Apple launches new AI-powered iPhone",
    "Opposition party wins snap election",
    "Bitcoin surges past $100,000",
    "Virat Kohli breaks Sachin's test record"
]

for headline in headlines:
    result = ask(few_shot_system, headline)
    print(f"{headline} → {result}")
