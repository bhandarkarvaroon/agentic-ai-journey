# day40.py — LangSmith tracing

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY")[:10] if os.getenv("LANGCHAIN_API_KEY") else "NOT SET")
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

print("\nCalling LLM...")
response = llm.invoke([{"role": "user", "content": "What is a deductible? One sentence."}])
print(f"Response: {response.content}")
print("\nCheck LangSmith for traces.")
