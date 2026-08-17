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
Developed by: VARSHA S
RegisterNumber: 212225040482

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):

    # Add bias column
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta
    theta = np.zeros((X.shape[1], 1))

    # Gradient Descent
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta = theta - learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta


# Load dataset
data = pd.read_csv("50_Startups.csv")

# Select R&D Spend as input and Profit as output
X1 = data[["R&D Spend"]].values
y = data[["Profit"]].values

# Standardize the input feature
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)

# Train the model
theta = linear_regression(X1_scaled, y)

# Display the parameters
print("Theta:")
print(theta)

# Predict Profit
X = np.c_[np.ones(len(X1_scaled)), X1_scaled]
predictions = X.dot(theta)

# Display Actual and Predicted values
results = pd.DataFrame({
    "Actual Profit": y.flatten(),
    "Predicted Profit": predictions.flatten()
})

print("\nActual vs Predicted Profit:")
print(results)
 
*/
```

## Output:


<img width="259" height="752" alt="Screenshot 2026-08-17 210204" src="https://github.com/user-attachments/assets/203bd11e-d74a-49a6-b386-e0abda27176e" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
