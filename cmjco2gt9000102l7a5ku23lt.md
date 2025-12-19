---
title: "The Mathematical Anatomy of Vectors"
datePublished: Tue Dec 02 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjco2gt9000102l7a5ku23lt
slug: the-mathematical-anatomy-of-vectors
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/_zsL306fDck/upload/68fea30ce14a7e820a2cc47fe6d04c2d.jpeg
tags: machine-learning, mathematics, linear-algebra, learning-journey, learning-in-public

---

I used to think of a Matrix as just a spreadsheet ,a grid of rows and columns. But as I dig deeper into the math, I realized that is a limited view.

To really understand Machine Learning, we have to look at the atomic unit: the **Vector**. Today is about pure Linear Algebra definitions, precise notation, and the axioms that govern how these numbers interact.

## **1\. Scalars and Vectors**

* **Scalar:** A single real number (c ∈ ℝ). It represents a magnitude, like a temperature, a price, or a weighting factor.
    
* **Vector:** An ordered list of scalars. If the list has n elements, we say the vector belongs to the space .
    

### **Standard Notation: The Column Vector**

In formal Linear Algebra, unless specified otherwise, a vector a is always a **vertical column**.

$$\vec{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}$$

Here, a₁, a₂, ..., aₙ are the individual **scalar components**.

Matrices as vectors

We can define a Matrix A not just as a grid of scalars, but as a horizontal collection of vectors.

If we have a set of vectors a₁, a₂, ..., aₙ , we can stack them side-by-side to form a Matrix A.

$$A = \begin{bmatrix} | & | & & | \\ \vec{a}_1 & \vec{a}_2 & \dots & \vec{a}_n \\ | & | & & | \end{bmatrix}$$

**Why does this distinction matter?**

* **Micro View:** We look at a vector v and see a stack of scalars (a₁, a₂, ..., aₙ).
    
* **Macro View:** We look at a Matrix A and see a row of vectors (a₁, a₂, ..., aₙ).
    

This allows us to treat a dataset not as a wall of text, but as a series of distinct feature vectors standing next to each other.

### **Special Types of Vectors**

In our vector space **ℝⁿ**, a few specific vectors have special roles.

**The Zero Vector (0)**

This is the "origin" of our space. It acts as the additive identity.

$$\vec{0} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

**Unit Vectors (û)**

A unit vector is any vector with a length (magnitude) of exactly 1. These are used to define pure direction.

The "Standard Basis Vectors" are special unit vectors that point along the x, y, z axes:

$$\vec{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \vec{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \vec{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

### **Sparse Vectors**

In high-dimensional spaces (like language processing), we often encounter **Sparsity**. A sparse vector is one where **most elements are zero**.Efficient algorithms ignore the zeros to save computation time.

$$\vec{v}_{sparse} = [0, 0, 5, 0, 0, 0, 2, 0]^T$$

### **High Dimensions (ℝⁿ)**

We live in 3D space ({R}³).

* ℝ: A line.
    
* ℝ²: A flat plane (like this screen).
    
* ℝ³: The physical world.
    

In Data Science, we work in ℝⁿ, where n can be thousands or millions. While we cannot visualize a 1,000-dimensional arrow, the **math works exactly the same way**. The rules of addition and scaling in 2D apply perfectly to 1000D.

## **2.Scalar Multiplication**

This is the simplest operation in Linear Algebra, but it has immense practical power.

Given a vector v and a scalar c, scalar multiplication scales the magnitude of the vector without changing its direction.

* If c &gt; 1, the vector stretches.
    
* If 0 &lt; c &lt; 1, the vector shrinks.
    
* If c is negative, the direction flips.
    

$$c \cdot \vec{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}$$

## **3.**Properties of Vector Addition

Just like 1 + 2 = 3, vectors have arithmetic rules. However, because they have both magnitude and direction, we must formally define their behavior to ensure our algorithms are stable.

If we have vectors 𝐮, 𝐯, 𝐰 and a zero vector 𝟎:

**A. Commutative Property**

Order doesn't matter. Walking North then East gets you to the same spot as walking East then North.

$$\vec{u} + \vec{v} = \vec{v} + \vec{u}$$

**B. Associative Property**

Grouping doesn't matter. You can add the first two, then the third, or the last two, then the first.

$$(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$$

**C. Additive Identity**

Adding the zero vector changes nothing.

$$\quad \vec{u} + \vec{0} = \vec{u}$$

**D. Additive Inverse**

For every vector u , there exists a negative vector -u (pointing in the exact opposite direction). Adding them together returns you to the origin.

$$\vec{u} + (-\vec{u}) = \vec{0}$$

## **4.Span of Vectors**

If I have a set of vectors, what are *all* the possible locations I can reach by combining them?

Definition:

The span of a set of vectors 𝐕 = {𝐯₁, ..., 𝐯ₖ} is the set of all possible linear combinations of those vectors. It is written as Span}(V).

### Example in ℝ²

Consider two vectors:

$$\vec{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \vec{v}_2 = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

Since these vectors point in different directions, their span covers **all of R²**. Any point (x, y) on a 2D plane can be reached by a specific combination of 𝐯₁ and 𝐯₂

The Exception:

If 𝐯₂ was just a multiple of 𝐯₁ (e.g., \[2, 4\]ᵀ), they would lie on the same line. Their span would just be that line, not the whole plane.

## **5\. Linear Combinations**

This is arguably the most important concept in Linear Algebra. It is the mechanism we use to build complex data from simple parts.

A linear combination of a set of vectors 𝐚₁, ..., 𝐚ₘ using scalars β₁, ..., βₘ is the new vector 𝐯 formed by adding the scaled versions together:

$$\vec{v} = \beta_1\vec{a}_1 + \beta_2\vec{a}_2 + \dots + \beta_m\vec{a}_m$$

The scalars βᵢ are called the coefficients of the linear combination.

**Unit Vectors as Building Blocks**

Any vector 𝐛 in n-dimensions can be expressed as a linear combination of the standard unit vectors (𝐞₁, ..., 𝐞ₙ).

For example, if 𝐯 = \[3, 5\]ᵀ, I can write it as:

$$\vec{v} = 3\begin{bmatrix}1\\0\end{bmatrix} + 5\begin{bmatrix}0\\1\end{bmatrix}$$

Here, the coefficients are just the entries of the vector itself.

## **6\. Linear Independence**

This concept is crucial because it tells us if our dataset contains "redundant" information.

**Definition:** A set of vectors is **linearly independent** if **no** vector in the set can be written as a linear combination of the others.

If you have a vector pointing North, and another pointing North-East, they are independent. But if you add a third vector that is just the sum of the first two, it adds no new directional information—it is **dependent**.

Vectors 𝐯₁, ..., 𝐯ₙ are linearly independent if and only if the only solution to the equation: (c₁ = c₂ = ... = 0).

$$c_1\vec{v}_1 + c_2\vec{v}_2 + \dots + c_n\vec{v}_n = \vec{0}$$

is the "trivial solution" where **all scalars are zero**.

* Logic: If you can find non-zero scalars that make the sum zero (e.g., 1𝐚 - 2𝐛 = 0), it proves that the vectors can cancel each other out—meaning they are merely scaled versions of each other (redundant).
    

## **5\. Length and Dot Product**

Finally, we need to measure "how big" a vector is. We formally call this the **Magnitude** or **Length** (often called the Norm). It is deeply connected to the **Dot Product**.

The dot product of a vector v with itself gives the square of its length:

$$\vec{v} \cdot \vec{v} = ||\vec{v}||^2$$

### **Geometric Interpretation**

In 2D space, for a vector 𝐯 = \[x, y\]ᵀ, this relationship is actually just the Pythagorean Theorem in disguise.

If we break the vector down:

1. **Dot Product:**
    

$$\vec{v} \cdot \vec{v} = (x \cdot x) + (y \cdot y) = x^2 + y^2$$

2. Length: By taking the square root, we get the familiar distance formula:
    

$$||\vec{v}|| = \sqrt{x^2 + y^2}$$