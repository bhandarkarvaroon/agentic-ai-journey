# day10.py - Type hints with the typing module

from typing import Optional, Union, List, Dict, Callable

# Optional - value can be type Or None

def get_bio(username: str) -> Optional[str]:
    bios = {"varun" : "Software Engineer", "rahul" : None}
    return bios.get(username)

print(get_bio("varun")) #Software Engineer
print(get_bio("rahul")) #None
print(get_bio("xyz"))   #None

#Union - value can be of multiple types
def process_input(value: Union[str, int]) -> str:
    return f"Received: {value}"

print(process_input("hello"))
print(process_input(42))

# List and Dict
def summerize_engineers(engineers : List[Dict[str, Union[str,int]]]) -> str:
    return f"Total engineers: {len(engineers)}"

engineers = [
    {"name" : "Varun", "years":10},
    {"name" : "Rahul", "years":5}
]

print(summerize_engineers(engineers))

# Callable - a parameter that is a function

def apply(func: Callable[[int, int], int], a: int, b:int) -> int:
    return func(a,b)

def multiply(a:int, b:int) -> int:
    return a*b

print(apply(multiply,3,4)) #12
print(apply(lambda a, b: a+b, 3,4)) #7