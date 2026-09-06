# day26.py - LangChain Expression Language (LCEL)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# -- Part 1 : Basic chain with lambda step --------

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant"),
    ("user","{input}")
])

# RunnableLambda wraps any python function into a chain step
uppercase = RunnableLambda(lambda x: x.upper())

chain = prompt | llm | parser | uppercase

response = chain.invoke({"input" : "What is RAG in AI? One sentence."})
print(response)

# --- Part 2: RunnableParallel - two chain at once ------
simple_prompt = ChatPromptTemplate.from_messages([
    ("system","Explain in simple terms of a beginner"),
    ("user","{input}")
])

technical_prompt = ChatPromptTemplate.from_messages([
    ("system","Explain technically for a software engineer"),
    ("user","{input}")
])

parallel_chain = RunnableParallel(
    simple = simple_prompt | llm | parser,
    technical = technical_prompt | llm | parser
)

results = parallel_chain.invoke({"input": "What is a vector database?"})

print("=== Simple ===")
print(results["simple"])
print("\n=== Technical ===")
print(results["technical"])