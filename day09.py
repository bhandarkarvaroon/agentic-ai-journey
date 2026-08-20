# day09.py - Decorators and context managers

import time
from typing import Callable

def timer(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def logger(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        print(f"Calling:{func.__name__}")
        result = func(*args, **kwargs)
        print(f"Result:{result}")
        print(f"Finished:{func.__name__}")
        return result
    return wrapper

class FileManager:
    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode

    def __enter__(self) -> object:
        self.file = open(self.filename, self.mode)
        print("File opened")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.file.close()
        print("File closed")

with FileManager("test.txt", "w") as f:
    f.write("Hello from context manager!")

