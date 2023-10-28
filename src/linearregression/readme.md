# Linear Regression with Gradient Descent

This repository contains a simple implementation of Linear Regression using Gradient Descent in Python. Linear Regression is a supervised machine learning algorithm used for predicting a continuous outcome variable (also called the dependent variable) based on one or more predictor variables (independent variables).

## Introduction

Linear Regression attempts to find the best-fitting linear relationship between the input features (X) and the target variable (y). This implementation uses Gradient Descent, an optimization algorithm, to iteratively adjust the model parameters and minimize the cost function.

## Mathematics Behind the Code

### 1. Hypothesis

The hypothesis function ($\hat{y}$) represents the predicted output based on the current weights ($w$).

$$\hat{y} = X \cdot w$$

### 2. Cost Function

The cost function (or loss function) measures the difference between the predicted values ($\hat{y}$) and the actual values (y). In this implementation, Mean Squared Error (MSE) is used as the cost function.

$$C = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

### 3. Gradient Descent

Gradient Descent is an iterative optimization algorithm that minimizes the cost function. The weights ($w$) are updated in the direction that reduces the cost.

$$\frac{\partial C}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

$$w = w - \alpha \frac{\partial C}{\partial w}$$


Where:
- $\alpha$ is the learning rate.
- $X$ is the matrix of input features.
- $\hat{y}$ is the predicted output.

### 4. Normal Equation

In addition to Gradient Descent, this implementation also includes the Normal Equation for finding the optimal weights directly.

$$w = (X^T X)^{-1} X^T y$$

## Usage

To use the Linear Regression model, follow these steps:

1. Import the `LinearRegression` class from `linear_regression.py`.
2. Create an instance of the `LinearRegression` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
from linear_regression import LinearRegression
import numpy as np

# Example Usage
X = np.random.rand(500, 1)
y = 3 * X + 5 + np.random.randn(500, 1) * 0.1

regression = LinearRegression(learning_rate=0.01, total_iterations=1000)
predictions = regression.fit_predict(X, y)
