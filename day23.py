# day23.py - PromptTemplates with multiple variables

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Template with multiple variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Answer in {language}"),
    ("user", "{question}")
])

chain = prompt | llm | parser

# Same chain, different inputs
print(chain.invoke({
    "domain": "insurance",
    "language": "simple English",
    "question": "What is a premium?"
}))

print("\n---\n")

print(chain.invoke({
    "domain": "Python programming",
    "language": "technical terms",
    "question": "What is a decorator?"
}))

print("\n---\n")

print(chain.invoke({
    "domain": "Financial Markets",
    "language": "simple English",
    "question": "What is a stock market index?"
}))

# Partial template — fix some variables, leave others open
partial_prompt = prompt.partial(
    domain="insurance",
    language="simple English"
)

# Now only need to pass the question
insurance_chain = partial_prompt | llm | parser

questions = [
    "What is a claim?",
    "What is a beneficiary?",
    "What is a co-pay?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {insurance_chain.invoke({'question': q})}\n")
