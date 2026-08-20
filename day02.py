class Engineer:
    def __init__(self, name: str, years: int, skills: list):
        self.name = name
        self.years = years
        self.skills = skills

    def introduce(self) -> str:
        return f"I'm {self.name} with {self.years} years of experience"

    def add_skill(self, skill:str) -> None:
        self.skills.append(skill)

    def is_senior(self) -> bool:
        return self.years >= 7

    def top_skill(self) -> str:
        return self.skills[0]

# Create 2 instances

varun = Engineer("Varun", 10, ["C#", "TypeScript"])
colleague_first = Engineer("Rahul", 5, ["Python","Django"])
colleague_second = Engineer("Saurabh", 15, ["Angular","Mongo"])

engineers = [varun, colleague_first, colleague_second]

print(varun.introduce())
print(varun.is_senior())
print(varun.top_skill())

print(colleague_first.introduce())

varun.add_skill("LangChain")
print(varun.skills)

areAllSeniors = True

if all(e.is_senior() for e in engineers):
    print("All engineers are seniors")
else:
    print("Not all engineers are seniors")