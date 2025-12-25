---
title: "Probability Distributions in ML"
datePublished: Thu Dec 11 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjl8horz000802jvc8x5adam
slug: probability-distributions-in-ml
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/XIIsv6AshJY/upload/2dd8a56c06e319a87810a9e0eb0440a9.jpeg
tags: machine-learning, probability, learning-journey, learn-in-public, probability-distributions

---

In Machine Learning, nothing is certain.When my model looks at a photo of a cat, it doesn't say "This is a Cat." It says: "I am 92% confident this is a cat, given the pointy ears and whiskers."

To become a Machine Learning Engineer, I had to rewire my brain from **Deterministic Thinking** to **Probabilistic Thinking**. Here are the core concepts I learned today the definitions, the distributions, and the theorems that govern how models think.

### **1\.** Probability Distribution

A probability distribution is a mathematical function that assigns probabilities to different outcomes. Unlike a simple frequency count, a distribution models the theoretical likelihood of future events.

* **Random Variables:**
    
    * **Discrete:** Specific, countable values (e.g., Is this email Spam? `0` or `1`).
        
    * **Continuous:** Infinite possibilities within a range (e.g., The probability that the temperature is exactly 72.0001°F).
        
* Independence:
    
    Two events are independent if the outcome of one does not influence the other.
    
    * Algorithms like **Naive Bayes** assume all features are independent to simplify calculations (even if they aren't!).
        
* **Probability Functions:**
    
    * **PMF (Probability Mass Function):** Used for Discrete data (e.g., rolling a die).
        
    * **PDF (Probability Density Function):** Used for Continuous data (e.g., height of a person).
        
* Expected Value (E\[X\]):
    
    The theoretical average outcome. In ML, our Loss Function is essentially trying to minimize the expected error over time.
    

### **2\. Critical Theorems**

These theorems allow models to reason about the world and update their predictions.

#### **Conditional Probability (P(A|B))**

This measures the probability of event A happening, **given** that we already know event B happened.

* *Example:* What is the probability this image is a Cat (A) given it has pointy ears (B)?
    

#### **Bayes' Theorem**

This is the formula for updating your beliefs based on new evidence. It is the engine behind Naive Bayes classifiers and many generative models.

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

#### **The Chain Rule**

This calculates the probability of a sequence of events.

* This is exactly how LLMs (like GPT) work. They compute the probability of the next word given the sequence of all previous words.
    

### **3\. The Essential Distributions**

ML models usually assume your data follows a specific shape. Here are the most common ones I've encountered.

#### **A. Discrete Distributions**

1\. Bernoulli Trials

A single experiment with two outcomes: Success (p) or Failure (1-p).

* *Example:* A single coin toss.
    

2\. Binomial Distribution

Models the number of successes (x) in n independent Bernoulli trials.

$$P(X=x) = ^nC_x p^x (1-p)^{n-x}$$

> **Example:** If I toss a fair coin 10 times (n=10, p=0.5), what is the chance of getting exactly 6 heads?

3\. Negative Binomial Distribution

Unlike the standard Binomial (fixed trials), this models the number of trials (n) needed to get a fixed number of successes (k).

$$P(X=n) = ^{n-1}C_{k-1} p^k (1-p)^{n-k}$$

> **Example:** Suppose the chance of finding a coupon in a pizza box is 30% (p=0.3). What is the probability I need to buy exactly 10 pizzas (n=10) to find my 3rd coupon (k=3)?

4\. Poisson Distribution

Models the frequency of an event occurring within a fixed interval of time or space.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

* *(*λ *is the average rate)*
    
    > **Example:** A server gets an average of 5 requests per second (λ=5). The chance of getting exactly 3 requests in the next second is approx 14%
    

#### **B. Continuous Distributions**

Continuous distributions apply to random variables with uncountable outcomes, such as height, time, or temperature.

#### **1\. Uniform Distribution**

In a uniform distribution, all outcomes in an interval \[a, b\] are equally likely.

**Probability Density Function (PDF):**

$$f(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \le x \le b \\ 0 & \text{otherwise} \end{cases}$$

**Cumulative Distribution Function (CDF):**

$$F(x) = \frac{x-a}{b-a}$$

**2\. Normal (Gaussian) Distribution**

This is the most critical distribution in Machine Learning. It models data that clusters symmetrically around a mean (μ), forming a bell shape.

* **PDF Formula:**
    

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

* ![Image of standard normal distribution curve](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcR05c81wEcLVuPEEZ2O-YAdB-FHSWqOOjC7TN1feo4wD6K1jvsV7wrsMCnPBOQOc0PjH11-5uR_NDKW3tWId0VWS9QXm8ZCGW9MfRX0xDcL1XPgYKQ align="left")
    

**3\. Chi-Square Distribution**

The Chi-Square χ² distribution is positively skewed and non-negative. It is primarily used in hypothesis testing to determine goodness-of-fit or independence.

* **Degrees of Freedom (k):** The number of independent values that can vary. For a contingency table,
    
    k = (Rows - 1) x (Columns - 1).
    
* **Mean:** k
    
* **Variance:** 2k**.**
    

**4.Measures of Uncertainty**

* Variance (σ²) & Standard Deviation (σ):
    
    These measure how "spread out" or risky the data is.
    
    * **High Variance:** The data is noisy and unpredictable.
        
    * **Low Variance:** The data is clustered tight (consistent).
        
* Covariance & Correlation:
    
    How two variables move together.
    
    > **Pro Tip:** If two features are 99% correlated (like "Temp in Celsius" and "Temp in Fahrenheit"), you should delete one. It adds no new information and slows down training.
    

### **4\. Advanced Concepts**

If you have ever wondered where Loss Functions actually come from or why we use specific metrics, the answer usually lies in Information Theory and Probabilistic Estimation.

#### **A. Maximum Likelihood Estimation (MLE)**

This is the foundational philosophy of training. When we train a model, we don't know the "true" parameters (weights) of the universe. We only have our data.

**The Goal:** Find the specific parameters (θ) that make our observed data **most probable**.

* *Translation:* "Adjust the weights until the model says, 'Yes, seeing this dataset is extremely likely.'"
    

The Problem: Probabilities are tiny numbers (e.g., 0.0001). Multiplying millions of them (to get total probability) results in underflow (the computer rounds down to zero).

The Fix: We take the Logarithm. This turns multiplication into addition, which computers handle much better. This is why we minimize "Log-Loss" instead of raw probability error.

$$\sum_{i=1}^{n} \log P(x_i | \theta)$$

#### **B. Entropy & Cross-Entropy**

These concepts come from Information Theory (Claude Shannon). They measure uncertainty, surprise, and information content.

1\. Entropy (H)

Entropy measures the amount of "disorder" or uncertainty in a distribution.

* *Low Entropy:* A coin that always lands on Heads (0% surprise).
    
* *High Entropy:* A fair coin toss (Maximum unpredictability).
    

$$H(P) = - \sum_{x} P(x) \log P(x)$$

2\. Cross-Entropy Loss (H(P, Q))

This is the standard loss function for Classification tasks. It measures the difference between the True Distribution (P) (the actual label, e.g., \[1, 0\]) and the Predicted Distribution (Q) (what the model thinks, e.g., \[0.9, 0.1\]).

We want to minimize this difference.

$$H(P, Q) = - \sum_{x} P(x) \log Q(x)$$

#### **C. KL Divergence (Kullback-Leibler Divergence)**

Think of this as the "distance" between two probability distributions. It tells us how much information is **lost** when we use the predicted distribution (Q) to approximate the true distribution (P).

* *If P and Q are identical:* KL Divergence = 0.
    
* *Application:* This is heavily used in **Generative AI** (like VAEs and GANs) to ensure the generated images follow the same statistical distribution as real images.
    

$$D_{KL}(P || Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$