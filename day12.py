# day12.py - JSON handling and Pydantic models

import json
from pydantic import BaseModel, ValidationError
from typing import Optional

# -- Part 1 : Raw JSON handling -------------

json_string = '{"name": "Varun", "years":10, "skills":["C#", "Python"]}'

# Parse JSON string to dict
data = json.loads(json_string)
print(data["name"])
print(data["skills"])

# Convert dict back to JSON string
back_to_json = json.dumps(data, indent=2)
print(back_to_json)


# ── Part 2: Pydantic models ───────────────────────────────────────────────
class Engineer(BaseModel):
    name: str
    years: int
    skills: list[str]
    bio: Optional[str] = None  # optional with default None

# Create from dict
engineer = Engineer(name="Varun", years=10, skills=["C#", "Python"])
print(engineer.name)
print(engineer.years)
print(engineer.bio)  # None

# Pydantic validates types automatically
try:
    bad_engineer = Engineer(name="Rahul", years="not_a_number", skills=["Python"])
except ValidationError as e:
    print(f"Validation error: {e}")

# Parse from dict directly
data = {"name": "Saurabh", "years": 15, "skills": ["Angular"]}
saurabh = Engineer(**data)
print(saurabh)
