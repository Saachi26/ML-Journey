---
title: "Linear Regression"
datePublished: Sat Jan 03 2026 11:39:54 GMT+0000 (Coordinated Universal Time)
cuid: cmjy8dkhp000002l4attsedr9
slug: linear-regression
tags: python, machine-learning, learning, scikit-learn, learning-journey, linearregression, learn-in-public

---

Linear Regression sounds fancy, but it is literally just the equation of a line I learned in 9th grade:

$$y = mx + b$$

* **y (Target):** What we want to predict (Price).
    
* **x (Feature):** The input data (Size).
    
* **m (Slope/Weight):** How much the price goes up for every extra sq ft.
    
* **b (Intercept/Bias):** The starting price (the base price of a house with 0 sq ft).
    

#### **The "Learning" Part (Cost Function)**

The computer doesn't know m or b. So how does it find them?

It uses a method called Ordinary Least Squares (OLS).

1. It draws a random line.
    
2. It measures the distance (error) between the line and every single data point.
    
3. It squares those errors and adds them up.
    
4. It rotates the line slightly to see if the total error goes down.
    

The "Best Fit Line" is simply the line where the **Total Error is at its minimum**.

### **2\. Simple Linear Regression (One Variable)**

Let's start with a simple synthetic dataset: **Years of Experience vs. Salary**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate Fake Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # Experience (0-2 years)
y = 4 + 3 * X + np.random.randn(100, 1) # Salary

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. View the "Learned" Parameters
print(f"Slope (m): {model.coef_[0][0]:.2f}")
print(f"Intercept (b): {model.intercept_[0]:.2f}")
# Expected: Slope ~3, Intercept ~4
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767440134970/8e370692-5354-41a4-a46d-0f8eddbbd469.png align="center")

### **3\. The Reality: Multiple Linear Regression**

In the real world, a house price isn't just based on **Size**. It's based on **Size**, **Bedrooms**, **Age**, and **Location**.

This is where the math changes from a Line to a Plane (or Hyperplane).

$$y = m_1x_1 + m_2x_2 + m_3x_3 + b$$

The amazing thing about Scikit-Learn? **The code is exactly the same.** You just feed it a matrix of features instead of a single column.

Python

```python
# Create complex data: 3 Features (Size, Bedrooms, Age)
X_complex = np.random.rand(100, 3) 
y_complex = 10 + 5 * X_complex[:, 0] + 3 * X_complex[:, 1] + 2 * X_complex[:, 2] + np.random.randn(100)

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_complex, y_complex, test_size=0.2, random_state=42)

# Train (Identical code!)
multi_model = LinearRegression()
multi_model.fit(X_train_c, y_train_c)

# Predict
predictions = multi_model.predict(X_test_c)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767440147224/7f386bc5-77ea-49bf-a12a-7bf2b47a509e.png align="center")

### **4\. Evaluation: How wrong are we?**

We can't just say "it looks good." We need metrics.

A. Mean Squared Error (MSE)

We take the error, square it (to remove negatives), and average it.

* *Low MSE = Good.*
    

B. R-Squared Score (R²)

How much of the "variance" did we explain?

* *1.0 = Perfect.*
    
* *0.0 = The model is just guessing the average.*
    

```python
print(f"R2 Score: {r2_score(y_test_c, predictions):.2f}")
# Output: 0.94 (Very good!)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767440296100/dd4f67a5-dd0e-421c-aa34-bb8ebaa6788d.png align="center")

### **5\. Visualizing the Truth**

For Simple Regression, we can plot the line. For Multiple Regression, we can't (it's 4D). Instead, we plot **Actual vs. Predicted**. If the model is perfect, all dots will land on the diagonal line.

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test_c, predictions, color='blue', alpha=0.5)
plt.plot([y_test_c.min(), y_test_c.max()], [y_test_c.min(), y_test_c.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Truth vs Prediction")
plt.show()
```

### **The Output**

I started with random noise. I ended with a mathematical function that can predict a salary based on 3 different inputs.

**Key Takeaway:** Linear Regression is the workhorse of Data Science. It’s fast, interpretable, and for many business problems, it’s all you need.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# PART 1: SIMPLE LINEAR REGRESSION (1 Variable)
# Goal: Visualize the "Line of Best Fit"
# ---------------------------------------------------------
print("--- PART 1: Simple Linear Regression ---")

# 1. Generate Synthetic Data (Experience vs Salary)
np.random.seed(42) # Ensure we get same random numbers every time
X = 2 * np.random.rand(100, 1)  # 0 to 2 years experience
y = 4 + 3 * X + np.random.randn(100, 1) # Salary = 4 + 3*Exp + Noise

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. View Learned Parameters (y = mx + b)
print(f"Learned Slope (m): {model.coef_[0][0]:.2f}")     # Expected ~3
print(f"Learned Intercept (b): {model.intercept_[0]:.2f}") # Expected ~4

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Visualize the Line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Prediction Line')
plt.title("Simple Linear Regression: Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# ---------------------------------------------------------
# PART 2: MULTIPLE LINEAR REGRESSION (3 Variables)
# Goal: Handle complex data (Size + Bedrooms + Age)
# ---------------------------------------------------------
print("\n--- PART 2: Multiple Linear Regression ---")

# 1. Generate Complex Data (3 Features)
# 100 samples, 3 features (e.g., Size, Bedrooms, Age)
X_complex = np.random.rand(100, 3) 
# The "True" Math: y = 10 + 5*x1 + 3*x2 + 2*x3 + Noise
y_complex = 10 + 5 * X_complex[:, 0] + 3 * X_complex[:, 1] + 2 * X_complex[:, 2] + np.random.randn(100)

# 2. Split Data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_complex, y_complex, test_size=0.2, random_state=42)

# 3. Train Model
multi_model = LinearRegression()
multi_model.fit(X_train_c, y_train_c)

# 4. Predict
predictions_c = multi_model.predict(X_test_c)

# 5. Evaluate Metrics
mse = mean_squared_error(y_test_c, predictions_c)
r2 = r2_score(y_test_c, predictions_c)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f} (1.0 is perfect)")

# 6. Visualize: Actual vs Predicted
# Since we can't plot 4D dimensions, we plot the accuracy of predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test_c, predictions_c, color='green', alpha=0.6)

# Draw a diagonal line (Perfect Prediction Line)
plt.plot([y_test_c.min(), y_test_c.max()], [y_test_c.min(), y_test_c.max()], 'r--', lw=2)

plt.title("Multiple Regression: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
```