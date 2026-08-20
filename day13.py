# day13.py - FastAPI with 2 endpoints

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Endpoint 1: GET /health
@app.get("/health")
def health() -> dict:
    return {"status" : "ok"}

# Endpoint 2: POST /greet
class GreetRequest(BaseModel):
    name: str

@app.post("/greet")
def greet(request: GreetRequest) -> dict:
    return {"message" : f"Hello, {request.name}!"}