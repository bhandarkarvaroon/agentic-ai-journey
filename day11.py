# day11.py - Environment variables and secrets management

from  dotenv import load_dotenv
import os

load_dotenv()

# Access variables

api_key = os.getenv("OPENAI_API_KEY")
environment = os.getenv("ENVIRONMENT","development") #default value if not set


print(f"Environment: {environment}")
print(f"API Key present: {api_key is not None}")

# Fail loudly if a required key is missing
def get_required_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

api_key = get_required_env("OPENAI_API_KEY")
print(f"Got API key: {api_key[:8]}...")  # print only first 8 chars — never print full keys
