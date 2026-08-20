from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

# Request model

class EngineerRequest(BaseModel):
    name : str
    years : int
    skills : list[str]
    bio: Optional[str] = None

# Response model

class EngineerReponse(BaseModel):
    name : str
    years : int
    skills : list[str]
    bio: str
    is_senior: bool

@app.post("/profile", response_model=EngineerReponse)
def create_profile(engineer: EngineerRequest) -> EngineerReponse:
    return EngineerReponse(
        name=engineer.name,
        years=engineer.years,
        skills=engineer.skills,
        bio=engineer.bio or "No bio available",
        is_senior=engineer.years >=7
    )