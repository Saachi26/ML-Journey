---
title: "The Unified Theory of Matrices & Vector Spaces"
seoDescription: "From linear systems to SVD mastering the architecture of data."
datePublished: Fri Dec 05 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjcstv7y000202lb4nz609pd
slug: the-unified-theory-of-matrices-and-vector-spaces
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/05A-kdOH6Hw/upload/46bc8ed2fd14005a75d927c6dfaea9d8.jpeg
tags: machine-learning, learning, mathematics, matrix, learning-journey, learning-in-public

---

## **1\. Anatomy of a Linear System**

Before we draw a matrix, we have to understand where it comes from: **Systems of Linear Equations**.

A generic linear system with m equations and n unknowns takes the form:

$$\begin{align*} a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n &= b_1 \\ a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n &= b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n &= b_m \end{align*}$$

**Deciphering the Labels** aᵢⱼ

The coefficients aᵢⱼ aren't random. The indices tell you exactly where they live:

* First Index (i): Represents the Row (Equation number).
    
* Second Index (j): Represents the Column (Variable number associated with xⱼ).
    

For example, a₂₃ is the coefficient in the 2nd equation for the 3rd variable.

### **Homogeneous vs. Non-Homogeneous**

There is a special name for systems based on the constants on the right side (b):

* **Homogeneous System**: If all constant terms are zero (b₁ = b₂ = ... = 0). The system essentially equals the zero vector 𝟎.
    
* **Non-Homogeneous System:** If at least one constant term bᵢ is not zero.
    

## **2\. Special Types of Matrices**

When we convert these systems into Matrix form, we encounter a few "celebrity" matrices.

1. **Zero Matrix (0):** A matrix where every single entry is 0. It acts as the "Additive Identity" (like the number 0).
    

$$\mathbf{0} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

2. Identity Matrix (I): A square matrix with 1s on the main diagonal and 0s everywhere else. It acts as the "Multiplicative Identity" (like the number 1).
    

$$I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

3. **Diagonal Matrix:** Non-zero values only appear on the main diagonal ($a\_{11}, a\_{22}, \\dots$). The rest are zero.
    

$$D = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}$$

4. **Ones Matrix:** A matrix filled entirely with 1s.
    

$$J = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

## **3\. Matrix Operations**

### **Scalar Multiplication**

Just like with vectors, we can scale a matrix by a single number. We simply multiply **every entry** aᵢⱼ by the scalar.

### **Matrix Multiplication**

Multiplying two matrices is **not** just multiplying their matching numbers. It's a series of dot products.

Definition:

To calculate the product C = A × B:

* Matrix A must be size m × n.
    
* Matrix B must be size n × p (The inner dimensions n must match!).
    
* The result C will be size m × p.
    

The Rule:

The entry cᵢⱼ is the dot product of the i-th row of A and the j-th column of B.

$$c_{ij} = \text{Row}_i(A) \cdot \text{Col}_j(B)$$

## **4\. Algebraic Laws (And a Warning)**

Matrices follow similar laws to real numbers, but with one massive exception.

* **Associative Law:** (AB)C = A(BC)
    
* **Distributive Law:** A(B + C) = AB + AC
    
* **Scalar Multiplication Law:** k(AB) = (kA)B = A(kB)
    
* **Commutative Law for Addition:** A + B = B + A
    

**No Commutativity for Multiplication**

In general, Order Matters.

$$A \times B \neq B \times A$$

Transforming data A by B is not the same as transforming data B by A.

## **5\. The Determinant**

The determinant is a special scalar value calculated from a **square matrix**.

### **What does it represent?**

Geometrically, the determinant tells us how much the matrix **scales** space:

* In 2D: It represents the **Area** spanned by the vectors.
    
* In 3D: It represents the **Volume** enclosed by the vectors.
    
* If det(A) = 0, the matrix squishes space into a lower dimension (area becomes a line, volume becomes a plane).
    

### **How to Calculate (3x3)**

For a 3 × 3 matrix, we use a formula involving the top row (a₁₁, a₁₂, a₁₃) and smaller 2 × 2 "sub-matrices":

$$\text{det}(A) = a_{11}\text{det} \begin{pmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{pmatrix} - a_{12}\text{det} \begin{pmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{pmatrix} + a_{13}\text{det} \begin{pmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{pmatrix}$$

## **6.Basis: Defining a Coordinate System**

A **Vector Space** is just a collection of vectors that we can add and scale. To navigate this space efficiently, we need a **Basis**.

Definition:

A Basis of a vector space is a specific set of vectors that has two properties:

1. **Linear Independence:** No vector in the set is redundant (you can't build one from the others).
    
2. **Span:** You can reach *every* point in the space by combining these vectors.
    

Example in ℝ²:

We usually use the standard axes 𝐞₁ = \[1, 0\]ᵀ and 𝐞₂ = \[0, 1\]ᵀ.

But any two vectors {𝐚, 𝐛} can form a basis for 2D space, as long as they don't point along the same line.

$$Basis = \{\vec{a}, \vec{b}\}$$

$$\text{Span}(\{\vec{a}, \vec{b}\}) = \mathbb{R}^2$$

## **7\. Projections: The Shadow of Data**

Sometimes, a vector lives in a high-dimensional space, but we want to approximate it in a lower-dimensional space (e.g., compressing data). We do this using **Projections**.

Definition:

The projection of a vector 𝐚 onto a line (defined by vector 𝐛) is the closest point on that line to 𝐚. Geometrically, it looks like dropping a perpendicular shadow.

The Formula:

To find the projection of 𝐚 onto 𝐛 (denoted proj\_𝐛 𝐚), we find a scalar c such that the "error vector" (𝐚 - c𝐛) is perpendicular (orthogonal) to 𝐛.

$$\text{proj}_{\vec{b}}\vec{a} = \frac{\vec{a} \cdot \vec{b}}{||\vec{b}||^2} \vec{b}$$

This calculation is the core of **Least Squares Regression** i.e., fitting a line to data points by minimizing the "error" distance.

## **8\. Orthogonality**

Two vectors are orthogonal if they are perpendicular to each other (they form a 1$90^\\circ$ angle).2 In high-dimensional space, we can't always "see" the angle, so we rely on the Dot Product test.

Vectors 𝐮 and 𝐯 are orthogonal if and only if their dot product is zero:

$$\vec{u} \cdot \vec{v} = 0$$

Why does this happen?

Recall that 𝐮 ⋅ 𝐯 = ‖𝐮‖ ‖𝐯‖ cos(θ).Since cos(90°) = 0, the entire product becomes zero.

**Key Property:**

If a set of non-zero vectors are mutually orthogonal, they are automatically Linearly Independent. You cannot build one perpendicular vector using the others.

## **9\. Orthonormal Bases**

Not all bases are created equal. The "best" kind of coordinate system is an **Orthonormal Basis**.

Definition:

A basis is orthonormal if it satisfies two conditions:

1. Orthogonal: All vectors are perpendicular to each other (Dot product is 0).
    
    $$\vec{a} \cdot \vec{b} = 0 \quad$$
    
2. Normal: All vectors have a unit length of 1.
    
    $$\quad ||\vec{a}|| = ||\vec{b}|| = 1$$
    

Why do we care?

Using an orthonormal basis simplifies calculations immensely. If our basis vectors are orthonormal, finding a projection is as simple as taking a dot product, with no messy division required.

## **10\. The Gram-Schmidt Process**

What if we have a "messy" basis and want to turn it into a clean "Orthonormal" one? We use an algorithm called **Gram-Schmidt**.

The Algorithm:

It works step-by-step to straighten out the vectors:

1. Normalize the first vector 𝐚₁ to get 𝐞₁.
    

Subtract the Projection: Take the next vector 𝐚₂ and subtract the part of it that points in the direction of 𝐞₁. This leaves only the perpendicular part.

$$\vec{v}k = \vec{a}k - \sum{i=1}^{k-1} \text{proj}{\vec{e}_i}(\vec{a}_k)$$

1. Normalize that remainder to get 𝐞₂.
    
2. **Repeat** for all vectors.
    

## **11\. Special Matrices**

In Linear Algebra, two types of matrices appear constantly because of their helpful properties.

* **Symmetric Matrix:** A square matrix that is equal to its own transpose (A = Aᵀ). The top-right mirrors the bottom-left.
    
* **Orthogonal Matrix:** A square matrix whose columns are orthonormal vectors.
    
    * Key Property: Its transpose is equal to its inverse!
        
        $$Q^T = Q^{-1}$$
        
        This makes solving equations with orthogonal matrices incredibly fast (O(n²) instead of O(n³)).
        

## **12\. Matrix Factorization (Decomposition)**

Just as we factor numbers 12 = 3 x 4 to understand their components, we factor matrices to reveal their hidden structure. This is essential for simplifying complex calculations and solving systems efficiently.

Here are the four "Big Guns" of Matrix Factorization:

### **A. QR Decomposition**

This method is closely related to the Gram-Schmidt process we just learned. It breaks a matrix down into a clean orthogonal component and a triangular component.

The Math: It decomposes a matrix A into:

$$A = Q R$$

* **Q (Orthogonal Matrix):** A matrix with orthonormal columns.
    
* **R (Upper Triangular Matrix):** A matrix where all entries below the main diagonal are zero.
    
* **Why it matters:**
    
    * It provides **numerically stable solutions** for linear systems (computers love it because it reduces rounding errors).
        
    * It is the standard way to solve **Linear Least Squares** problems (finding the best fit line).
        
    * Used extensively in signal processing.
        

### **B. LU Decomposition**

If you need to solve a system of linear equations (Ax = b), you usually reach for LU Decomposition. It separates the matrix into "Lower" and "Upper" parts.

The Math: It decomposes a matrix A into:

$$A = L U$$

* **L (Lower Triangular):** Entries above the diagonal are zero.
    
* **U(Upper Triangular):** Entries below the diagonal are zero.
    
* **Why it matters:**
    
    * It facilitates **matrix inversion** and solving linear equations efficiently.
        
    * It is incredibly common in engineering and physical sciences.
        

### **C. Singular Value Decomposition (SVD)**

If Linear Algebra had a "King," it would be SVD. Unlike other methods that require square matrices, SVD works on **any** matrix. It is the foundation of data compression.

The Math: It decomposes a matrix into three specific matrices:

$$A = U \Sigma V^*$$

* **U:** An **Orthogonal** matrix.
    
* Σ\*\*:\*\* A **Diagonal** matrix containing the "singular values" (magnitudes).
    
* **V\*:** The **Conjugate Transpose** of an orthogonal matrix.
    
* **Why it matters:**
    
    * It is the engine behind **Principal Component Analysis (PCA)** used for dimensionality reduction.
        
    * Used for **Data Compression** (images) and **Noise Reduction**.
        
    * It reveals the true Rank and structure of a matrix.
        

### **D. Eigen Decomposition**

This decomposition reveals the "fundamental properties" of a matrix by finding the vectors that don't change direction when the matrix is applied to them.

The Math: It breaks a matrix down into Eigenvalues (λ) and Eigenvectors(**𝐯**).

$$A\vec{v} = \lambda\vec{v}$$

* **Why it matters:**
    
    * Critical for analyzing **stability** in systems of differential equations.
        
    * The basis for AI algorithms involved in **Feature Extraction**.
        
    * Used to understand linear transformations geometrically (stretching vs. rotating).
        

## **Conclusion: Choosing the Right Tool**

To choose the right tool for the job.

* Need to solve a square system? **LU Decomposition**.
    
* Need to compress a rectangular dataset? **SVD**.
    
* Need to fix a skewed coordinate system? **Gram-Schmidt**.
    

Understanding these structures allows us to manipulate high-dimensional data as easily as we manipulate 2D arrows on paper.