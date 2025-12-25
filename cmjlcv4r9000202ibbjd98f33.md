---
title: "Pandas: A Deep Dive"
datePublished: Tue Dec 23 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjlcv4r9000202ibbjd98f33
slug: pandas-a-deep-dive
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/8qEB0fTe9Vw/upload/7f14e1219663332b49c5b66b11931ee9.jpeg
tags: python, machine-learning, learning, pandas, learning-journey, learn-in-public

---

### 1\. Pandas Series

A `Series` is a one-dimensional array holding data of any type. It’s essentially a single column.

#### Creating Labels

By default, if I make a list, Pandas labels it 0, 1, 2... just like a normal array. But I can define my own **Labels**. This is crucial , it lets me access data by *meaning*, not just position.

```python
import pandas as pd

a = [1, 7, 2]

# Default Indexing
my_var = pd.Series(a)
print(my_var[0]) # Output: 1

# Custom Labels
my_var = pd.Series(a, index=["x", "y", "z"])
print(my_var["y"]) # Output: 7
```

#### Key/Value Objects as Series

Since a Series maps a Label to a Value, it behaves exactly like a Python Dictionary. In fact, you can create a Series directly from a Key/Value object.

```python
calories = {"day1": 420, "day2": 380, "day3": 390}

my_series = pd.Series(calories)
print(my_series)
# Output:
# day1    420
# day2    380
# ...
```

### 2\. Pandas DataFrames

A DataFrame is a 2-dimensional data structure. It’s a collection of Series sharing the same index.

#### Locate Row (`loc`)

This confused me at first. In a Python list, we use `list[0]`. In a DataFrame, we use the `.loc[]` attribute to locate one or more rows.

```python
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data)

# Return row 0 as a Series
print(df.loc[0])

# Return row 0 AND 1 as a DataFrame
print(df.loc[[0, 1]])
```

#### Named Indexes

Just like with Series, we can name the rows in a DataFrame. This turns our random numbers into a meaningful lookup table.

```python
# Giving the rows names instead of 0, 1, 2
df = pd.DataFrame(data, index=["day1", "day2", "day3"])

# Locate Named Indexes
print(df.loc["day2"])
# Output: 
# calories    380
# duration     40
```

### 3\. Loading Data: Reading Files

We usually pull data from external files.

#### Read CSV Files

CSV (Comma Separated Values) is the most common format.

```python
df = pd.read_csv('data.csv')

# If you print the whole DF, Pandas truncates the middle rows!
print(df)
```

#### The `max_rows` Setting

I ran into an issue where I wanted to see *everything*, but Pandas hid the middle rows with `...`. You can check and modify the system's maximum row display limit.

```python
# Check current limit (usually 60)
print(pd.options.display.max_rows) 

# Increase the limit to see ALL rows
pd.options.display.max_rows = 9999
print(df)
```

#### Read JSON

JSON (JavaScript Object Notation) is standard for Big Data sets.

* **Dictionary as JSON:** If your JSON code is just a Python Dictionary, Pandas can read it directly without a file.
    

```python
data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103
  }
}

df = pd.DataFrame(data)
print(df)
```

```python
# JSON often comes as a list of dictionaries
# Pandas handles the conversion automatically
df_json = pd.read_json('data.json')

print(df_json.to_string()) # .to_string() prints the entire dataframe
```

**Pro Tip:** If your JSON is nested (dictionaries inside dictionaries), you might need `pd.json_normalize()`.

### 4\. Analyzing the Data

Once the file is loaded, how do we know if it's broken?

#### Viewing the Data

Instead of printing the whole thing (and freezing my VS Code), I use:

* `head(10)`: First 10 rows.
    
* `tail(10)`: Last 10 rows.
    

#### Info About the Data ([`df.info`](http://df.info)`()`,)

This is the single most useful command in Pandas. It tells you the schema.

and `describe()` instantly calculates statistics for every numerical column.

**Output usually looks like this:**

* **Count:** How many rows?
    
* **Mean:** The average.
    
* **Std:** Standard Deviation (how spread out is the data?).
    
* **Min/Max:** The range boundaries.
    
* **25%/50%/75%:** The quartiles.
    

```python
print(df.info())
print(df.describe())
```

**Result Explained:**

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 169 entries, 0 to 168
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Duration  169 non-null    int64  
 1   Pulse     169 non-null    int64  
 2   Maxpulse  169 non-null    int64  
 3   Calories  164 non-null    float64
dtypes: float64(1), int64(3)
memory usage: 5.4 KB
```

1. **RangeIndex:** Tells you how many entries (rows) you have.
    
2. **Data columns:** How many columns.
    
3. **Non-Null Count:** *This is critical.* If you have 169 rows, but the "Calories" column says "164 non-null", you know you have 5 missing values (Null Values).
    
4. **Dtype:** Tells you if the data is an integer (`int64`), float, or object (text).
    

#### Finding Relationships (Correlation)

Does the duration of the workout correlate with calories burned?

```python
print(df.corr())
# Output ranges from -1 to 1. 
# 1.0 means perfect correlation.
```

#### Null Values

[`df.info`](http://df.info)`()` reveals Null values, but `df.isnull()` finds them. Handling these empty cells is the first step of "Data Cleaning," which I'll tackle tomorrow.

```python
# Returns a massive table of True/False
print(df.isnull())

# A better way: Count the errors
print(df.isnull().sum())
```

```plaintext
Duration    0
Pulse       0
Maxpulse    0
Calories    5  <-- There they are!
dtype: int64
```