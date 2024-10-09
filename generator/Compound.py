
from typing import List
from generator.Function import Function

# Define the abstract class Compound that implements Function
class Compound(Function):
    def __init__(self):
        self.sub_functions: List[Function] = []  # Initialize the list to hold Function objects

    def add_function(self, f: Function):
        """Add a function to the list."""
        self.sub_functions.append(f)