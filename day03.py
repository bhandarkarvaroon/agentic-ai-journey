from dotenv import load_dotenv
import os

load_dotenv()

name = os.getenv("MY_NAME")
role = os.getenv("MY_ROLE")

print(f"Loaded from .env: {name} is a {role}")


# Write to a file
with open("engineers.txt","w") as f:
    f.write("Varun,10,C#\n")
    f.write("Rahul,5,Python\n")
    f.write("Saurabh,15,Angular\n")

# Read it back
with open("engineers.txt", "r") as f:
    content = f.read()

print(content)

# Count lines
with open("engineers.txt","r") as f:
    lines = f.readlines()

print(f"Total engineers: {len(lines)}")

for line in lines:
    data = line.strip().split(",")
    engineer = {"name": data[0],"years": data[1], "skill":data[2]}
    print(f"{engineer['name']} knows {engineer['skill']}")