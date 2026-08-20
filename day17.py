# day17.py - Streaming responses from OpenAI

from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    {"role": "system", "content":"You are a helpful assistant."},
    {"role": "user", "content":"Explain what an API is in 3 sentences"}
]

print("Response:",end="",flush=True)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
        time.sleep(0.1)  # 50ms delay per token

print() # new line at the end