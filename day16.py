# day16.py - Multi turn chat with conversation history

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    {"role" : "system", "content" : "You are a helpful assistant for a software engineer learning python and AI."}
]

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content":user_input})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )

    print("Assistant: ", end="", flush=True)
    reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            reply += token

    messages.append({"role": "assistant", "content": reply})
    print("\n")
