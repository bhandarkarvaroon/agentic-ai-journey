# Variables and types
name = "Varun"
age = 33
is_engineer = True

# Function
def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet(name))

# List
skills = ["C#", "TypeScript", "Angular"]
for skill in skills:
    print(skill)


#Dict
profile = {"name": "Varun", "years": 10, "role": "Manager"}
print(profile["name"])
print(profile["years"])
print(profile["role"])

def summarise(profile: dict) -> str:
    return f"{profile["name"]} is a {profile["role"]} with {profile["years"]} years of experience"

print(summarise(profile))