# day39.py — Sequential chains with LCEL

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# ── Chain 1: Extract key insurance terms from a document ─────────────────
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an insurance expert. Extract the 3 most important terms from the given policy text."),
    ("user", "Policy text: {policy_text}")
])
extract_chain = extract_prompt | llm | parser

# ── Chain 2: Explain the extracted terms in simple language ───────────────
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly insurance agent explaining terms to a first-time buyer. Be simple and clear."),
    ("user", "Explain these insurance terms in simple language: {terms}")
])
explain_chain = explain_prompt | llm | parser

# ── Chain 3: Generate a customer-friendly summary ─────────────────────────
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service agent. Write a friendly 2-sentence summary."),
    ("user", "Based on these explanations, write a summary: {explanations}")
])
summary_chain = summary_prompt | llm | parser

# ── Full sequential pipeline ──────────────────────────────────────────────
full_chain = (
    {"terms": extract_chain}
    | RunnablePassthrough.assign(explanations=explain_chain)
    | {"explanations": lambda x: x["explanations"]}
    | summary_chain
)

# Test
policy_text = """
This auto insurance policy includes a $500 deductible for collision coverage.
The policyholder pays a monthly premium of $150. In case of an accident,
the insured must notify the insurer within 30 days. Coinsurance applies
at 80/20 split after the deductible is met. The out-of-pocket maximum
is $3,000 per year.
"""

print("=== Policy Text ===")
print(policy_text.strip())
print()

print("=== Step 1: Extracted Terms ===")
terms = extract_chain.invoke({"policy_text": policy_text})
print(terms)
print()

print("=== Step 2: Explanations ===")
explanations = explain_chain.invoke({"terms": terms})
print(explanations)
print()

print("=== Step 3: Customer Summary ===")
summary = summary_chain.invoke({"explanations": explanations})
print(summary)
