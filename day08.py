#day08.py - List comprehensions, generator, lambda functions

numbers = [1,2,3,4,5,6,7,8,9,10]

# Traditional loop
evens_loop = []
for n in numbers:
    if n%2 == 0:
        evens_loop.append(n)

#List comprehension - same result
evens_comp = [n for n in numbers if n%2 == 0]

print(evens_loop)
print(evens_comp)

# Squares of even numbers
squares = [n ** 2 for n in numbers if n%2 == 0]
print(squares)

skills = ["C#", "TypeScript", "Angular", "Python", "LangChain", "SQL"]

long_skills = [skill for skill in skills if len(skill) > 6]

print(long_skills)

engineers = [
    {"name": "Varun", "years": 10},
    {"name": "Rahul", "years": 5},
    {"name": "Saurabh", "years": 15},
]

# Sort by years using lambda
sorted_engineers = sorted(engineers, key=lambda e: e["years"])
for e in sorted_engineers:
    print(f"{e['name']} — {e['years']} years")

# Generator function
def fibonacci(n: int): 
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Print first 10 fibonacci numbers
for num in fibonacci(10):
    print(num, end=" ")
print()

# Generator expression (like list comprehension but with parentheses)
squares_gen = (n ** 2 for n in range(1, 6))
for s in squares_gen:
    print(s, end=" ")
print()
