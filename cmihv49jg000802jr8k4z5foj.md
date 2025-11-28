---
title: "Why I Ditched Google Colab for a Local Setup?"
datePublished: Thu Nov 27 2025 20:04:44 GMT+0000 (Coordinated Universal Time)
cuid: cmihv49jg000802jr8k4z5foj
slug: why-i-ditched-google-colab-for-a-local-setup
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1764273689937/8db0f826-5fcd-4d16-9ef0-3472fc009534.png
tags: python, machine-learning, python3, macos, numpy, anaconda

---

In my [Day 0 post](https://www.google.com/search?q=LINK_TO_PREVIOUS_POST&authuser=1), I committed to learning Machine Learning in public. The first hurdle wasn't the math,it was the **environment**.

As a beginner, the standard advice is usually: *"Just use Google Colab."* And honestly, Colab is great. It’s free, it lives in the browser, and it gives you access to a powerful GPU.

But I am a developer. I like owning my setup. I like using VS Code extensions, customizing my theme, and not having my runtime disconnect because I went to grab a coffee. Plus, I’m running an **M1 MacBook**, which means I have a surprisingly powerful Neural Engine sitting right under my keyboard.

Today, I’m documenting how I turned my Mac into a local Data Science workstation.

> **Note for Windows/Linux Users:** I’m documenting my setup on an **M1 MacBook Air**. If you are on Windows, the concepts (Miniconda, Environments, VS Code) are exactly the same, but the installation steps will differ slightly. I recommend following the [official Conda docs](https://docs.anaconda.com/miniconda/) for your OS, then come back here for the configuration!

### Step 1: The Manager : Miniconda

Coming from Web Development (Node.js), I’m used to `npm` handling my packages. In Python, things get messy quickly if you just `pip install` everything globally. You end up with "Dependency Hell"—where Project A needs `NumPy 1.2` but Project B needs `NumPy 2.0`.

To fix this, I’m using **Miniconda**. It’s a lightweight version of Anaconda that lets me create isolated "bubbles" (virtual environments) for different projects.

**The Installation:** I used Homebrew (because of course):

```bash
brew install --cask miniconda
conda init zsh
```

If you don't have Homebrew, you can either [install it](https://brew.sh/) or do it the **normal way** by downloading the installer directly from the [Miniconda website](https://docs.anaconda.com/miniconda/).

### Step 2: Creating the "Code Train Repeat" Environment

Instead of installing libraries into the void, I created a specific environment for my ML journey. This keeps my system Python clean and happy.

Create a new environment with Python 3.9 and then Activate it i.e Enter the bubble you just created.

```bash
conda create -n ml-journey python=3.9
conda activate ml-journey
```

Now, I installed the "Big Three" libraries I’ll need for the next month:

```bash
conda install numpy pandas matplotlib scikit-learn jupyter
```

### Step 3: The Editor (VS Code)

I can't imagine coding without **VS Code**. It’s lightweight, fast, and the extension ecosystem is unbeatable.

To make it ML-ready, I installed the **Jupyter Extension** by Microsoft. This is a game-changer. It allows me to run Jupyter Notebooks (`.ipynb` files) *directly* inside VS Code. I get the interactive cells of a notebook combined with the IntelliSense and Copilot features of a pro IDE.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764272891040/c8c16bc5-7458-434e-a620-0d097726a5b5.png align="center")

### Step 4: Verification

To make sure everything was actually working, I ran a quick script to test if my environment could handle a basic data plot.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot it
plt.plot(x, y)
plt.title("Sine Wave: Environment Check")
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1764272636291/510d38da-b522-4c63-9340-fc9d247885e8.png align="center")

If your window popped up with this wave, you’re golden. Our environment is live.

One major reason I chose local setup is Popular Deep Learning libraries like PyTorch and TensorFlow have been optimized for Mac. They can use the M1's GPU for training models, which is significantly faster than a standard CPU.

I’ll be diving deeper into this when I reach the Deep Learning module of my roadmap.

### Status Update

* **Environment:** ✅ Ready
    
* **IDE:** ✅ Configured
    
* **Motivation:** 📈 High
    

Next up, I’m diving into **Linear Algebra** to understand the math that powers these libraries. See you in the next log!