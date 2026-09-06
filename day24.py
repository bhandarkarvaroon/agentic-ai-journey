# day24.py - Output parsers

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# -- Part 1: StrOutputParser (you already know this) -----------
str_parser = StrOutputParser()

# -- Part 2: JsonOutputParser ---------------------
json_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant.Always respond in valid JSON."),
    ("user", "Give me 3 insurance policy types with name and description. Return as JSON array.")
])

json_chain = json_prompt | llm | JsonOutputParser()
result = json_chain.invoke({})
print(type(result)) # should be list, not string
print(result)


# -- Part 3 : PydanticOutputParser ----------------
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class PolicyType(BaseModel):
    name: str = Field(description="Name of the insurance policy")
    description: str = Field(description="Brief description")
    is_mandatory: bool = Field(description="Whether this insurance is legally mandatory")

class PolicyList(BaseModel):
    policies: List[PolicyType] = Field(description="List of insurance policies")

pydantic_parser = PydanticOutputParser(pydantic_object=PolicyList)

pydantic_prompt = ChatPromptTemplate.from_messages([
    ("system","You are an insurnace expert"),
    ("user","List 3 insurnace types.\n{format_instructions}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())

pydantic_chain = pydantic_prompt | llm | pydantic_parser
result2 = pydantic_chain.invoke({})

for policy in result2.policies:
    print(f"{policy.name} - Mandatory: {policy.is_mandatory}")
    print(f" {policy.description}\n")
          