---
title: "Python Basics for Machine Learning"
datePublished: Wed Dec 17 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjlbnwgi000302l4cv5m3g9h
slug: python-basics-for-machine-learning
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/QUwM2LDVs3A/upload/cc43a477b5c4c92d2e8e4077b893ef6a.jpeg
tags: python, machine-learning, basics, learning-journey, learning-in-public

---

Today, I’m documenting the essential Python syntax for ML—from simple printing to functional programming.

### **1\. Formatting Output**

The `print()` function is the first thing we learn, but formatting variables is where it gets useful.

**The** `.format()` Method This allows us to inject variables into strings. We can even assign keywords to avoid order confusion.

```python
name = 'John Doe'
age = 25

# Basic formatting
print('My name is {} and I am {} years old'.format(name, age))

# Keyword formatting (Safer)
print('My name is {one} and I am {two} years old'.format(two=age, one=name))
```

> While `.format()` is common, modern Python (3.6+) often uses **f-strings**: `print(f"My name is {name}")`. I'm keeping both in my toolkit.

### **2\. Data Structures:**

Python has four built-in data structures. Confusing them causes errors, so I’m breaking down their core properties: **Ordered**, **Changeable (Mutable)**, and **Allow Duplicates**.

#### A. Lists

Lists are the most versatile collection. They use square brackets `[]`.

* **Ordered:** When we say lists are ordered, it means the items have a defined index (0, 1, 2...). If you add a new item, it goes to the end. That order *will not change* unless you explicitly move things.
    
* **Changeable (Mutable):** You can change, add, or remove items *after* the list is created.
    
* **Allow Duplicates:** Since items are found by index, not value, you can have `[1, 1, 1]` without issues.
    

```python
vowels = ['a', 'e', 'I', 'o', 'u']

# Appending: Adding to the end
vowels.append('a') 
# Result: ['a', 'e', 'I', 'o', 'u', 'a'] (Duplicates allowed!)

# Removing: Deleting specific items
vowels.remove('e')

# Nesting: Lists inside lists
nest = [1, 2, [3, 4]]
print(nest[2][1]) # accessing the '4' inside the inner list
```

#### B. Tuples

Tuples use parentheses `()` and look like lists, but they behave differently.

* **Ordered:** Like lists, they have a defined order and index.
    
* **Immutable (Unchangeable):** This is the key difference. Once created, you cannot change, add, or remove items.
    
* **Why use them?** They are faster than lists and "write-protected." Perfect for data that shouldn't be touched (like coordinates or database records).
    

```python
# A tuple representing a single record
student = ('Ajika', 21, 'CS Major')

# Tuple Unpacking (Crucial for iterating data!)
x = [(1,2), (3,4), (5,6)]

for a, b in x:
    print(f"First: {a}, Second: {b}")
# This splits the pairs automatically.
```

#### C. Sets

Sets use curly braces `{}`. Think of them like a mathematical bag of concepts.

* **Unordered:** The items have no index. You cannot ask for `my_set[0]` because the computer doesn't store them in a specific row.
    
* **Unchangeable (Sort of):** You cannot change an item (e.g., turn 'a' into 'b'), but you *can* add or remove items from the set.
    
* **No Duplicates:** This is their superpower.
    

```python
messy_data = {'e', 'a', 'e', 'I', 'o', 'u', 'I'}
print(messy_data)
# Output: {'e', 'a', 'I', 'o', 'u'} -> All duplicates vanished.
```

#### D. Dictionaries

Dictionaries store data in `key:value` pairs using `{}`.

* **Ordered (as of Python 3.7+):** They remember the insertion order.
    
* **Changeable:** You can change values (e.g., update a user's age).
    
* **No Duplicate Keys:** You cannot have two 'name' keys. If you add a second one, it overwrites the first.
    

```python
d = {
    'key1': 'value1',
    'key2': [1, 2, 3],  # You can store a list inside!
    'k3': {'inner_key': 100} # You can even nest dictionaries!
}

# Accessing nested data
print(d['k3']['inner_key']) # Output: 100
```

**Useful Dict Methods:**

* `.keys()`: Returns all keys.
    
* `.values()`: Returns all values.
    
* `.items()`: Returns pairs (useful for loops).
    

### **3\. Control Flow: Logic & Loops**

#### Logic in Python is defined by **indentation** (whitespace). There are no curly braces `{}` for code blocks here; you must respect the `Tab`.

**Conditions**

```python
a = 10
b = 20

if b > a:
    print("b is greater")
elif a == b:
    print("Equal")
else:
    print("a is greater")
```

#### **Loops**

* **For Loops:** Iterate over a sequence.
    
* **While Loops:** Run as long as a condition is true.
    

```python
# The Range function creates a sequence of numbers
# range(start, stop) - stop is exclusive!
for i in range(1, 5):
    print(i) 
# Output: 1, 2, 3, 4
```

### **4\. Functions: Reusable Logic**

A function is a block of code which only runs when it is called. It can return data as a result. It helps avoiding code repetition.

```python
def my_func(param1):
    """
    THIS IS A DOCSTRING.
    It explains what the function does.
    Check it by pressing Shift+Tab in VS Code/Jupyter!
    """
    print(param1)

my_func("Hello World")
```

#### **List Comprehensions**

The reverse of a for-loop. It condenses 3 lines of code into 1.

```python
x = [1, 2, 3, 4]

# Old way
out = []
for num in x:
    out.append(num**2)

# List Comprehension way
out = [num**2 for num in x] 
# Output: [1, 4, 9, 16]
```

#### **Lambda Expressions**

A lambda function is a small anonymous function.It can take any number of arguments, but can only have one expression.The power of lambda is better shown when you use them as an anonymous function inside another function.Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:

```python
# Instead of defining a whole function to multiply by 2...
t = lambda var: var * 2
print(t(6)) # Output: 12

def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
print(mydoubler(11))
```

#### **Map and Filter**

These functions enable efficient data transformation and processing by applying operations to entire iterables (like lists or tuples) without using explicit loops.

* **Map:** *map(fun, iter)* Applies a function to every item in a list.
    
    **Parameters**:
    
    * **fun**: It is a function to execute on each element of the iterable object.
        
    * **iter**: It is iterable to be reduced
        
    * **initial (optional)**: Initial value for the accumulator.
        
* **Filter:** *filter(function, sequence)* Keeps only items that match a condition.
    
    **Parameters:**
    
    * **function:** function that tests if each element of a sequence is true or not.
        
    * **sequence:** sequence which needs to be filtered, it can be sets, lists, tuples, or containers of any iterators.
        

```python
seq = [1, 2, 3, 4, 5]

# MAP: Multiply everything by 2
list(map(lambda num: num*2, seq)) 
# Output: [2, 4, 6, 8, 10]

# FILTER: Keep only even numbers
list(filter(lambda num: num%2 == 0, seq))
# Output: [2, 4]
```