#day22.py - First LangChain chain

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Model
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant."),
    ("user","{input}")
])

# 3. Output parser
parser = StrOutputParser()

# 4. Chain - pipe them together
chain = prompt | llm | parser

# 5. Invoke
response = chain.invoke({"input" : "What is LangChain in one sentence?"})
print(response)

# Second chain — insurance domain
insurance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert insurance assistant. Be concise."),
    ("user", "{input}")
])

insurance_chain = insurance_prompt | llm | parser

response2 = insurance_chain.invoke({"input": "What is a deductible?"})
print(response2)
