# day27.py — Q&A chain with structured Pydantic output using LangChain

from dotenv import load_dotenv
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class InsuranceRelatedAnswer(BaseModel):
    answer: str = Field(description="Answer")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level of the answer")
    follow_up_questions: list[str] = Field(description="Follow up questions")

pydantic_parser = PydanticOutputParser(pydantic_object=InsuranceRelatedAnswer)

pydantic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an insurance expert."),
    ("user", "Topic: {topic}\nQuestion: {question}\n{format_instructions}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())

chain = pydantic_prompt | llm | pydantic_parser

def print_result(result: InsuranceRelatedAnswer) -> None:
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print("Follow up questions:")
    for q in result.follow_up_questions:
        print(f"  - {q}")
    print()

# Test 1
result1 = chain.invoke({
    "topic": "auto insurance",
    "question": "What happens if I miss a premium payment?"
})
print_result(result1)

# Test 2
result2 = chain.invoke({
    "topic": "health insurance",
    "question": "What is the difference between HMO and PPO?"
})
print_result(result2)
