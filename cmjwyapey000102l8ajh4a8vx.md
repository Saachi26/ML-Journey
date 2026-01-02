---
title: "Data Wrangling — Cleaning, Merging, and Grouping"
datePublished: Fri Dec 26 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjwyapey000102l8ajh4a8vx
slug: data-wrangling-cleaning-merging-and-grouping
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/fyeOxvYvIyY/upload/02880d1abcf6584cf1e0b24bd7529c70.jpeg
tags: data, python, machine-learning, learning, pandas, machinelearning, learning-journey

---

I thought I just had to load a CSV and press "Train." I was wrong. My data was scattered across two different files. It had empty rows. It had text where numbers should be.

If I feed this into a model, it will crash. Today is about **Data Wrangling**: The art of Merging, Cleaning, and Shaping data.

### **1\. Assembling the Data: Merging & Concat**

Before I can clean, I need to get everything in one place.

#### **Scenario A: Stacking (**`pd.concat`)

I have `data_january.csv` and `data_february.csv`. They have the same columns, just different rows. I need to stack them on top of each other.

```python
df1 = pd.read_csv('jan_data.csv')
df2 = pd.read_csv('feb_data.csv')

# Axis 0 = Stack vertically (add rows)
full_df = pd.concat([df1, df2], axis=0)
```

#### **Scenario B: The Database Join (**`pd.merge`)

This is like a SQL Join. I have a **Users Table** (ID, Name) and a **Sales Table** (ID, Purchase). I need to connect them via the `ID`.

```python
users = pd.DataFrame({'ID': [1, 2], 'Name': ['Max', 'Jaden']})
sales = pd.DataFrame({'ID': [1, 2], 'Amount': [500, 700]})

# Merge on the 'ID' column
# 'inner' means only keep rows where ID exists in BOTH tables
merged_df = pd.merge(users, sales, on='ID', how='inner')
```

### **2\. Handling Empty Cells (**`NaN`)

"NaN" stands for Not a Number. It’s a hole in the dataset. We have two choices: Remove it or Fix it.

#### **Option A: The Nuclear Option (**`dropna`)

If a row is missing critical data, it's garbage. Throw it out.

```python
# Drop rows containing ANY missing values
new_df = df.dropna()

# Drop rows only if 'Calories' is missing
df.dropna(subset=['Calories'], inplace=True)
```

* **Pros:** 100% clean data.
    
* **Cons:** You might delete 50% of your dataset if you aren't careful.
    

#### **Option B: The Surgeon's Option (**`fillna`)

Instead of deleting, we make an educated guess. This is called **Imputation**. For numerical data, we usually fill the hole with the **Mean** (average) or **Median** (middle value).

```python
# Calculate the mean (average)
x = df["Calories"].mean()

# Fill empty cells with that average
df["Calories"].fillna(x, inplace=True)
```

* **Why Median?** If you have outliers (e.g., one person burned 10,000 calories), the Mean will be skewed. The Median is safer.
    

### **2\. Cleaning Wrong Formats**

Sometimes data exists, but it's in the wrong costume.

* **Dates as Text:** "2023/12/01" is a String, not a Date.
    
* **Numbers as Text:** "450" (string) cannot be multiplied.
    

```python
# Convert to DateTime format
df['Date'] = pd.to_datetime(df['Date'])

# Fix specific errors (e.g., typo in row 7)
df.loc[7, 'Date'] = '2023-12-15'
```

### **3\. Converting Text to Numbers**

This was the hardest concept for me to grasp. Machine Learning models is just math. You cannot multiply "Cat" \* 5. You have to turn "Cat" into a number.

#### **Why not just label them 0, 1, 2?**

If I have `Red`, `Green`, `Blue`, and I make them `0`, `1`, `2`:

* The model thinks `Blue (2)` is "greater than" `Red (0)`.
    
* It thinks `Green (1)` is the average of Red and Blue.
    
* **This implies a ranking that doesn't exist.**
    

#### **The Solution: One-Hot Encoding**

We create a new column for *every* category and mark it with a 1 or 0.

| **Color** | **Is\_Red** | **Is\_Green** | **Is\_Blue** |
| --- | --- | --- | --- |
| Red | 1 | 0 | 0 |
| Green | 0 | 1 | 0 |
| Blue | 0 | 0 | 1 |

```python
# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['Color'])

print(df.head())
# Now I have columns like 'Color_Red', 'Color_Green' with 1s and 0s.
```

> **Pro Tip:** In Scikit-Learn (which I'll use later), this is done using `OneHotEncoder`, but for data analysis, Pandas `get_dummies` is the quick fix.

### **4\. Removing Duplicates**

Sometimes you accidentally scrape the same data twice. Duplicate rows bias the model (it memorizes that specific example).

```python
# Check for duplicates
print(df.duplicated().sum())

# Remove them
df.drop_duplicates(inplace=True)
```

### **5\. Grouping (**`groupby`)

This is where Pandas becomes powerful. `groupby` allows us to split the data into categories, apply a function, and combine the results. It’s essential for analysis and **smarter cleaning**.

**Basic Analysis:** "What is the average calorie burn for each Workout Type?"

```python
# Group by 'Type' and calculate the mean of 'Calories'
print(df.groupby('Type')['Calories'].mean())
```

**Advanced Cleaning (Smart Imputation):** *The problem:* Filling a missing "Age" with the *global* average might be wrong. *The solution:* Fill missing "Age" with the average age **of that person's specific Job Title**.

```python
# Fill missing values based on group averages
df['Age'] = df['Age'].fillna(df.groupby('Job')['Age'].transform('mean'))
```

### **The Output**

My dataset started as a mess. Now:

1. **Merged** into a single Master DataFrame.
    
2. Empty cells are filled with the Median.
    
3. Dates are actual Date Objects.
    
4. **Imputed** missing values (using Group statistics).
    
5. "Categories" are now mathematically readable "One-Hot" vectors.
    
6. Duplicates are gone.