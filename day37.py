# day37.py — Auto-summarising chatbot for long conversations

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class SummaryChat:
    def __init__(self, summarise_every: int = 3):
        self.summarise_every = summarise_every
        self.summary = ""
        self.recent_messages = []
        self.turn_count = 0

    def _build_context(self) -> list:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"You are an insurance assistant. Previous conversation summary: {self.summary}"
            })
        else:
            context.append({
                "role": "system",
                "content": "You are an insurance assistant."
            })
        context.extend(self.recent_messages)
        return context

    def _summarise(self) -> None:
        if not self.recent_messages:
            return
        history = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in self.recent_messages
        ])
        prompt = f"Previous summary: {self.summary}\n\nNew conversation:\n{history}\n\nUpdate the summary in 3 sentences."
        response = llm.invoke([
            {"role": "system", "content": "You summarise conversations concisely."},
            {"role": "user", "content": prompt}
        ])
        self.summary = response.content
        self.recent_messages = []  # clear recent after summarising
        print(f"\n[Summary updated: {self.summary[:100]}...]\n")

    def chat(self, user_input: str) -> str:
        self.recent_messages.append({"role": "user", "content": user_input})
        context = self._build_context()
        response = llm.invoke(context)
        reply = response.content
        self.recent_messages.append({"role": "assistant", "content": reply})
        self.turn_count += 1

        # Summarise every N turns
        if self.turn_count % self.summarise_every == 0:
            self._summarise()

        return reply


# Test with 9 messages
chatbot = SummaryChat(summarise_every=3)

questions = [
    "What is a deductible?",
    "What is a premium?",
    "What is a copay?",
    "How does coinsurance work?",
    "What is an out-of-pocket maximum?",
    "What is a beneficiary?",
    "What did I ask about first?",
    "Can you summarise what we discussed?",
    "Which of these concepts is most important for a first-time insurance buyer?",
]

for q in questions:
    print(f"You: {q}")
    reply = chatbot.chat(q)
    print(f"Assistant: {reply[:150]}\n")
