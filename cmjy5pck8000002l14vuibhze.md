---
title: "Scikit-Learn & Linear Regression"
datePublished: Wed Dec 31 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjy5pck8000002l14vuibhze
slug: scikit-learn-and-linear-regression
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1767435893445/3fdd93a1-d589-45dd-9be5-802249a05ba4.jpeg
tags: python, machine-learning, learning, scikit-learn, learning-journey, learning-in-public

---

Scikit-Learn (`sklearn`) is the industry standard for classical Machine Learning. Whether you are classifying emails as spam or predicting house prices, this library handles the heavy lifting.

Today, I’m unpacking the toolkit to understand exactly how it works.

### **1\. A Consistent Interface**

The genius of Scikit-Learn isn't just the algorithms; it's the **design**. Whether you are building a simple Regression or a complex Decision Tree, the code looks *exactly the same*.

This means I only have to learn three commands:

1. [`model.fit`](http://model.fit)`(X, y)`: "Hey model, here is the data. Learn the patterns." (Training)
    
2. `model.predict(X_new)`: "Here is new data. What is the answer?" (Inference)
    
3. `model.score(X, y)`: "How well did you do?" (Evaluation)
    

### **2\. What Can It Do?**

Scikit-Learn divides Machine Learning into four main territories.

#### **A. Regression (Predicting a Quantity)**

* **The Goal:** Predict a continuous number (e.g., Stock Price, Temperature, Salary).
    
* **The Tools:** `LinearRegression`, `RandomForestRegressor`, `SVR`.
    

#### **B. Classification (Predicting a Label)**

* **The Goal:** Predict a category (e.g., Spam vs. Not Spam, Cat vs. Dog).
    
* **The Tools:** `LogisticRegression`, `KNeighborsClassifier`, `SVC`.
    

#### **C. Clustering (Grouping Data)**

* **The Goal:** The data has no labels. You want the machine to find natural groups automatically (e.g., "Customer Segments").
    
* **The Tools:** `KMeans`, `DBSCAN`, `SpectralClustering`.
    

#### **D. Dimensionality Reduction (Simplifying Data)**

* **The Goal:** You have too many columns (1000 features). You want to compress them into the 3 most important ones to visualize them.
    
* **The Tools:** `PCA` (Principal Component Analysis), `t-SNE`.
    

After manually cleaneing data using Pandas. It turns out, Scikit-Learn has professional tools to do this better, ensuring you treat your Training and Test data exactly the same way.

**Scaling Data (**`StandardScaler`) Neural Networks and distance-based models (like KNN) fail if your data is on different scales (e.g., Age is 0-100, but Salary is 0-100,000).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit on training data, transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Encoding Text (**`LabelEncoder` & `OneHotEncoder`) The official way to turn "Cat/Dog" into numbers.

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(["cat", "dog", "cat"]) 
# Output: [0, 1, 0]
```

### **3\. Model Selection: The Referee**

How do we know our model works? Scikit-Learn provides the judges.

**Splitting Data** The most famous function in the library.

```python
from sklearn.model_selection import train_test_split
```

**Metrics** It has a metric for every problem type.

* **Regression:** `mean_squared_error`
    
* **Classification:** `accuracy_score`, `confusion_matrix`
    

### **4\. The "Pro" Move: Pipelines**

This blew my mind. You can chain all these steps (Preprocessing -&gt; Scaling -&gt; Modeling) into a single object called a **Pipeline**.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Create a workflow
my_pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step 1: Scale data
    ('model', LinearRegression())      # Step 2: Train model
])

# Now I can just call .fit() on the pipeline!
my_pipeline.fit(X_train, y_train)
```

### **Example - Step 1: Loading the Data**

Scikit-Learn comes with "Toy Datasets" built-in, so we can practice without downloading CSVs. I used the famous **Iris Dataset** (classifying flowers).

```python
from sklearn.datasets import load_iris

# Load the data
iris = load_iris()
X = iris.data   # The Features (Input)
y = iris.target # The Labels (Output)

print("Features:", iris.feature_names)
# Output: ['sepal length', 'sepal width', 'petal length', 'petal width']
```

### **Step 2: Splitting the Data**

This is the Golden Rule of ML: **Never test on the data you trained on.** We split the data into a **Training Set** (to learn) and a **Testing Set** (to evaluate).

```python
from sklearn.model_selection import train_test_split

# Split: 60% for training, 40% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
```

### **Step 3: Preprocessing (The Tricky Part)**

Machine Learning models only understand numbers. They cannot read "Cat" or "Dog." We must convert text into numbers. Scikit-Learn gives us two tools for this:

#### **A. Label Encoding (**`LabelEncoder`)

Assigns a unique number to each category (Cat=0, Dog=1, Bird=2).

* *Best for:* Ordinal data (Low, Medium, High).
    

```python
from sklearn.preprocessing import LabelEncoder

data = ['cat', 'dog', 'dog', 'bird']
encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data)

print(encoded_data) 
# Output: [1 2 2 0]
```

#### **B. One-Hot Encoding (**`OneHotEncoder`)

Creates a new column for every category (Is\_Cat, Is\_Dog, Is\_Bird).

* *Best for:* Nominal data (colors, animals) where no order exists.
    

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Reshape because OneHotEncoder expects 2D array
data = np.array(['cat', 'dog', 'bird']).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
print(encoder.fit_transform(data))
# Output:
# [[1. 0. 0.]   <- Cat
#  [0. 1. 0.]   <- Dog
#  [0. 0. 1.]]  <- Bird
```

### **Step 4: Training & Prediction**

Now that the data is ready, we train the model. I chose **Logistic Regression**.

```python
from sklearn.linear_model import LogisticRegression

# 1. Instantiate the model
log_reg = LogisticRegression(max_iter=200)

# 2. Train the model (The "Fit")
log_reg.fit(X_train, y_train)

# 3. Make Predictions
y_pred = log_reg.predict(X_test)
```

### **Step 5: Evaluation**

Did it work? We compare our predictions (`y_pred`) against the actual answers (`y_test`).

```python
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

**Testing on New Data:** We can even feed it a completely new flower sample to see what it thinks:

```python
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
preds = log_reg.predict(sample)

# Convert number back to name
print("Predictions:", [iris.target_names[p] for p in preds])
# Output: ['virginica', 'virginica']
```

Scikit-Learn isn't just a library; it's a framework.

* **Preprocessing:** `LabelEncoder`, `OneHotEncoder`
    
* **Modeling:** `LogisticRegression`
    
* **Evaluation:** `accuracy_score`
    

It handles the math so I can focus on the logic.