---
title: "Deep Dive into Machine learning Math"
datePublished: Fri Dec 19 2025 06:59:12 GMT+0000 (Coordinated Universal Time)
cuid: cmjciqsyb000502ilea8le4qa
slug: deep-dive-into-machine-learning-math
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
    

P2D=(x,y)∈R2 P3D=(x,y,z)∈R3

### 3\. Angles & Measurement (Degrees vs. Radians)

An angle measures the amount of rotation between two lines.

* **Degrees:** The circle is divided into 360 slices.
    
* **Radians:** The standard unit in mathematics. One radian is the angle where the arc length equals the radius. A full circle is 2π radians (approx 6.28).
    

Radians=Degrees×π180

x=rcos⁡θ y=rsin⁡θ tan⁡θ=yx

### 4\. The Unit Circle

The Unit Circle is a circle with a radius of exactly 1, centered at (0,0). Any point on the circle edge can be defined by the angle The x-coordinate is the Cosine, and the y-coordinate is the Sine.

x=cos⁡θ y=sin⁡θ x2+y2=1

### 5\. Fundamental Trigonometric Identities

An Identity is an equation that is true for every possible value of the angle. I think of these as compression tools, they let us swap out messy, complex terms for simple ones.

A. The Pythagorean Identity

This is the most famous identity. It states that for any angle, the square of sine plus the square of cosine always equals 1.

* It acts as a "sanity check." It guarantees that our coordinates (x,y) stay on the Unit Circle (radius = 1). In Machine Learning, we use this to **normalize vectors** stripping away the magnitude so we can focus purely on direction.
    
* tanθ is how steep the hill is (Slope). secθ is the actual distance you walk up that hill (Hypotenuse). This identity lets us calculate the distance traveled just by knowing the steepness!
    

sin2⁡θ+cos2⁡θ=1 1+tan2⁡θ=sec2⁡θ

B. Sum & Difference Formulas:

These allow us to calculate the sine or cosine of angles that are added, subtracted, or doubled, without doing geometry from scratch.

* In data processing, we often rotate vectors. If I rotate a vector by angle α and then again by β, I don't want to calculate two separate rotations. I can use the sum formula sin(α + β) to compute the final position in one step.
    

sin⁡(α±β)=sin⁡αcos⁡β±cos⁡αsin⁡β cos⁡(α±β)=cos⁡αcos⁡β∓sin⁡αsin⁡β

C. Double Angle Formulas:

These are special cases of the Sum formulas where the two angles are identical (α = β). They express the sine or cosine of an angle's "double" (2θ) purely in terms of the original angle θ.

* Often in calculus or physics engines you end up with messy terms like 2sinθcosθ.This formula lets you compress that entire mess into a single, clean term: sin(2θ). It simplifies the math significantly.
    

sin⁡(2θ)=2sin⁡θcos⁡θ cos⁡(2θ)=cos2⁡θ−sin2⁡θ

### 6\. Law of Sines and Cosines

Standard trigonometry, only works for right-angled triangles. When we have a generic triangle, we use these laws to solve for missing sides or angles.

* **Law of Sines:** Proportional relationship between sides and angles.
    

asin⁡A=bsin⁡B=csin⁡C 

* **Law of Cosines:** A generalized version of Pythagoras for any triangle.
    

c2=a2+b2−2abcos⁡C

### 7\. Norms & Euclidean Distance

**Norm:** The size or length of a vector. It measures how far the point is from the origin (0,0).

|x|=∑i=1nxi2 

**Euclidean Distance:** The straight-line distance between two specific points in space. It is calculated using the Pythagorean theorem.

d(P,Q)=(q1−p1)2+⋯+(qn−pn)2