---
title: "Deep Dive into Machine learning Math"
datePublished: Sun Nov 30 2025 18:30:00 GMT+0000 (Coordinated Universal Time)
cuid: cmjciqsyb000502ilea8le4qa
slug: deep-dive-into-machine-learning-math
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/OPpCbAAKWv8/upload/de51bf21bf030d44408a1cef79a75584.jpeg
tags: machine-learning, mathematics, linear-algebra, learning-journey, learning-in-public

---

I am currently working my way through a massive **Linear Algebra for Machine Learning** course. To be honest, a lot of this felt like déjà vu. I technically learned about vectors, dot products, and the unit circle back in high school.

I realized I didn't need a textbook; I needed a **Cheat Sheet**. I needed a single place where all the scattered definitions, identities, lived together stripped of the fluff.

So, I wrote them down.

This article isn't a tutorial; it's my personal documentation. It covers the absolute essentials of **Vector Spaces**, **Trigonometry**, and **Euclidean Geometry** that serve as the bedrock for the algorithms I'm about to build.

### 1\. Real Numbers & Vector Spaces

Before we can do any math, we need to define the "space" we are working in.

* **Real Numbers {R}:** This represents the set of all continuous numbers - decimals, fractions, negatives, and zero (e.g., 3.14, -5, 0).
    
* **Vector Space:** This is just a fancy term for a collection of vectors that have n components. If a vector has 2 numbers (x, y), it lives in 2D space {R}². If it has 3 numbers, it lives in 3D space {R}³
    

$$\begin{aligned} &\mathbf{x} \in \mathrm{R}^n \\ &\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \end{aligned}$$

### 2\. The Cartesian Coordinate System

The Cartesian system is the classic grid we use to locate points. It uses two perpendicular number lines (axes) to specify a position.

* **The Origin:** The center point (0,0) where the axes intersect.
    
* **Coordinates:** An ordered pair (x, y)tells you exactly how far to move right/left (x-coordinate) and up/down (y-coordinate) from the origin.
    

$$\begin{aligned} &P_{2D} = (x, y) \in \mathrm{R}^2 \\ &P_{3D} = (x, y, z) \in \mathrm{R}^3 \end{aligned}$$

### 3\. Angles & Measurement (Degrees vs. Radians)

An angle measures the amount of rotation between two lines.

* **Degrees:** The circle is divided into 360 slices.
    
* **Radians:** The standard unit in mathematics. One radian is the angle where the arc length equals the radius. A full circle is 2π radians (approx 6.28).
    

$$\mathrm{Radians} = \mathrm{Degrees} \times \frac{\pi}{180}$$

$$\begin{aligned} &x = r \cos \theta \\ &y = r \sin \theta \\ &\tan \theta = \frac{y}{x} \end{aligned}$$

### 4\. The Unit Circle

The Unit Circle is a circle with a radius of exactly 1, centered at (0,0). Any point on the circle edge can be defined by the angle The x-coordinate is the Cosine, and the y-coordinate is the Sine.

$$\begin{aligned} &x = \cos \theta \\ &y = \sin \theta \\ &x^2 + y^2 = 1 \end{aligned}$$

### 5\. Fundamental Trigonometric Identities

An Identity is an equation that is true for every possible value of the angle. I think of these as compression tools, they let us swap out messy, complex terms for simple ones.

A. The Pythagorean Identity

This is the most famous identity. It states that for any angle, the square of sine plus the square of cosine always equals 1.

* It acts as a "sanity check." It guarantees that our coordinates (x,y) stay on the Unit Circle (radius = 1). In Machine Learning, we use this to **normalize vectors** stripping away the magnitude so we can focus purely on direction.
    
* tanθ is how steep the hill is (Slope). secθ is the actual distance you walk up that hill (Hypotenuse). This identity lets us calculate the distance traveled just by knowing the steepness!
    

$$\begin{aligned} &\sin^2 \theta + \cos^2 \theta = 1 \\ &1 + \tan^2 \theta = \sec^2 \theta \end{aligned}$$

B. Sum & Difference Formulas:

These allow us to calculate the sine or cosine of angles that are added, subtracted, or doubled, without doing geometry from scratch.

* In data processing, we often rotate vectors. If I rotate a vector by angle α and then again by β, I don't want to calculate two separate rotations. I can use the sum formula sin(α + β) to compute the final position in one step.
    

$$\begin{aligned} &\sin(\alpha \pm \beta) = \sin \alpha \cos \beta \pm \cos \alpha \sin \beta \\ &\cos(\alpha \pm \beta) = \cos \alpha \cos \beta \mp \sin \alpha \sin \beta \end{aligned}$$

C. Double Angle Formulas:

These are special cases of the Sum formulas where the two angles are identical (α = β). They express the sine or cosine of an angle's "double" (2θ) purely in terms of the original angle θ.

* Often in calculus or physics engines you end up with messy terms like 2sinθcosθ.This formula lets you compress that entire mess into a single, clean term: sin(2θ). It simplifies the math significantly.
    

$$\begin{aligned} &\sin(2\theta) = 2 \sin \theta \cos \theta \\ &\cos(2\theta) = \cos^2 \theta - \sin^2 \theta \end{aligned}$$

### 6\. Law of Sines and Cosines

Standard trigonometry, only works for right-angled triangles. When we have a generic triangle, we use these laws to solve for missing sides or angles.

* **Law of Sines:** Proportional relationship between sides and angles.
    

$$\begin{aligned} &\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} \\ \end{aligned}$$

* **Law of Cosines:** A generalized version of Pythagoras for any triangle.
    

$$\begin{aligned} &c^2 = a^2 + b^2 - 2ab \cos C \end{aligned}$$

### 7\. Norms & Euclidean Distance

**Norm:** The size or length of a vector. It measures how far the point is from the origin (0,0).

$$\begin{aligned} &\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{n} x_i^2} \\ \end{aligned}$$

**Euclidean Distance:** The straight-line distance between two specific points in space. It is calculated using the Pythagorean theorem.

$$\begin{aligned} d(P, Q) = \sqrt{(q_1 - p_1)^2 + \dots + (q_n - p_n)^2} \end{aligned}$$

That is the foundation. We have covered the spaces where data lives , the ways we measure it, and the rules that govern its geometry.

This page is meant to be bookmarked. It’s the reference manual.

Now that we can define *where* the data is, we need to learn how to move it. Next, we tackle **Calculus & Gradients**.