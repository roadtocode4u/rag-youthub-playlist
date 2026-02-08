# Python Programming Study Notes
## For Beginner Students

---

## Chapter 1: Introduction to Python

### What is Python?
Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability and simplicity.

### Key Features of Python
- Easy to learn and read
- Interpreted language (no compilation needed)
- Dynamically typed
- Object-oriented programming support
- Large standard library
- Cross-platform compatibility

### Python Versions
- Python 2 (legacy, ended support in 2020)
- Python 3 (current version, recommended)

---

## Chapter 2: Variables and Data Types

### Variables
A variable is a container for storing data values. In Python, you don't need to declare variable types.

Example:
```python
name = "Alice"  # string
age = 25        # integer
height = 5.6    # float
is_student = True  # boolean
```

### Basic Data Types

1. **Integer (int)**: Whole numbers without decimals
   - Examples: 1, 42, -17, 0

2. **Float**: Numbers with decimal points
   - Examples: 3.14, -0.5, 2.0

3. **String (str)**: Text enclosed in quotes
   - Examples: "Hello", 'Python', """Multi-line"""

4. **Boolean (bool)**: True or False values
   - Only two values: True, False

5. **List**: Ordered, mutable collection
   - Example: [1, 2, 3, "apple"]

6. **Tuple**: Ordered, immutable collection
   - Example: (1, 2, 3)

7. **Dictionary (dict)**: Key-value pairs
   - Example: {"name": "Alice", "age": 25}

---

## Chapter 3: Operators

### Arithmetic Operators
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Floor Division: `//`
- Modulus: `%`
- Exponentiation: `**`

### Comparison Operators
- Equal to: `==`
- Not equal to: `!=`
- Greater than: `>`
- Less than: `<`
- Greater than or equal: `>=`
- Less than or equal: `<=`

### Logical Operators
- and: Returns True if both conditions are true
- or: Returns True if at least one condition is true
- not: Reverses the boolean value

---

## Chapter 4: Control Flow

### If Statements
Used for conditional execution of code.

```python
if condition:
    # code block
elif another_condition:
    # code block
else:
    # code block
```

### For Loops
Used to iterate over sequences (lists, strings, etc.)

```python
for item in sequence:
    # code block
```

### While Loops
Repeats while a condition is true.

```python
while condition:
    # code block
```

### Break and Continue
- `break`: Exits the loop entirely
- `continue`: Skips to the next iteration

---

## Chapter 5: Functions

### What is a Function?
A function is a reusable block of code that performs a specific task.

### Defining Functions
```python
def function_name(parameters):
    # code block
    return value
```

### Parameters vs Arguments
- Parameters: Variables in function definition
- Arguments: Actual values passed when calling

### Return Statement
The `return` statement sends a value back to the caller.

### Lambda Functions
Anonymous, single-expression functions:
```python
square = lambda x: x ** 2
```

---

## Chapter 6: Lists

### Creating Lists
```python
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
```

### List Methods
- `append()`: Add item to end
- `insert()`: Add item at position
- `remove()`: Remove first occurrence
- `pop()`: Remove and return item
- `sort()`: Sort the list
- `reverse()`: Reverse the list
- `len()`: Get list length

### List Slicing
```python
list[start:end:step]
```

### List Comprehension
```python
squares = [x**2 for x in range(10)]
```

---

## Chapter 7: Dictionaries

### Creating Dictionaries
```python
student = {
    "name": "John",
    "age": 20,
    "courses": ["Math", "Science"]
}
```

### Dictionary Methods
- `keys()`: Get all keys
- `values()`: Get all values
- `items()`: Get key-value pairs
- `get()`: Get value safely
- `update()`: Update dictionary

### Accessing Values
```python
student["name"]  # Using key
student.get("name")  # Using get method
```

---

## Chapter 8: Error Handling

### Try-Except Block
```python
try:
    # code that might cause error
except ErrorType:
    # handle the error
finally:
    # always executes
```

### Common Exceptions
- `ValueError`: Invalid value
- `TypeError`: Wrong type
- `IndexError`: List index out of range
- `KeyError`: Dictionary key not found
- `ZeroDivisionError`: Division by zero

---

## Chapter 9: File Handling

### Opening Files
```python
file = open("filename.txt", "mode")
```

### File Modes
- `r`: Read (default)
- `w`: Write (overwrites)
- `a`: Append
- `r+`: Read and write

### With Statement (Recommended)
```python
with open("file.txt", "r") as file:
    content = file.read()
```

### File Methods
- `read()`: Read entire file
- `readline()`: Read one line
- `readlines()`: Read all lines as list
- `write()`: Write to file

---

## Chapter 10: Object-Oriented Programming (OOP)

### Classes and Objects
- Class: A blueprint for creating objects
- Object: An instance of a class

### Defining a Class
```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def bark(self):
        print(f"{self.name} says woof!")
```

### OOP Concepts
1. **Encapsulation**: Bundling data with methods
2. **Inheritance**: Creating new classes from existing ones
3. **Polymorphism**: Same method, different behaviors
4. **Abstraction**: Hiding implementation details

### The __init__ Method
The constructor method that initializes object attributes.

---

## Quick Reference Card

| Concept | Syntax |
|---------|--------|
| Print | `print("Hello")` |
| Input | `input("Enter: ")` |
| Comment | `# This is a comment` |
| Variable | `x = 10` |
| List | `[1, 2, 3]` |
| Dictionary | `{"key": "value"}` |
| If | `if x > 0:` |
| For loop | `for i in range(5):` |
| Function | `def func():` |
| Class | `class MyClass:` |

---

## Study Tips

1. Practice coding every day
2. Start with small programs
3. Read and understand error messages
4. Use comments to explain your code
5. Build projects to apply knowledge
6. Join coding communities
7. Don't be afraid to ask questions
