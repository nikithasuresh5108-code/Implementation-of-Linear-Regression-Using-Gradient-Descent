# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:nagalakshmi       
RegisterNumber: 25003017
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values

# ----- Feature scaling -----
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std

# Parameters
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []

# Gradient Descent
for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2 / n) * np.sum((y_hat - y) * x)
    db = (2 / n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y)

x_sorted = np.argsort(x)
plt.plot(
    x[x_sorted],
    (w * x + b)[x_sorted],
    color='red'
)

plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
<img width="1633" height="745" alt="Screenshot 2026-01-30 134550" src="https://github.com/user-attachments/assets/509bbd4c-bad0-4159-b326-740cf61ae26a" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
