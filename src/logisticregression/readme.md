# Logistic Regression with Gradient Descent
This repository contains a straightforward implementation of Logistic Regression using Gradient Descent in Python. Logistic Regression is a supervised machine learning algorithm primarily used for binary classification tasks, predicting the probability that an instance belongs to a particular class.

## Introduction
Logistic Regression aims to model the probability that an instance belongs to a particular class. This implementation employs Gradient Descent, an optimization algorithm, to iteratively update the model parameters and minimize the logistic loss function.

## Mathematics Behind the Code

### 1. Hypothesis

The hypothesis function ($\hat{y}$) represents the predicted output based on the current weights ($w$).

$$\hat{y} = \sigma(X \cdot w + b)$$

where $\sigma$, the sigmoid function, is defined as

$$\sigma(z) = \frac{1}{(1 + e^{-z})}$$

which outputs are between 0 and 1, representing the probability of the positive class.

### 2. Cost Function - Binary cross-entropy

The logistic loss function is used to measure the error between the predicted probabilities ($\hat{y}$) and the true labels ($y$):

$$C(w, b) = -\frac{1}{m} * \sum_{i=1}^{m} y^{(i)} * log(\hat{y}^{(i)}) + (1 - y^{(i)}) * log(1 - \hat{y}^{(i)})$$

where $m$ is the number of samples

### 3. Gradient Descent

The weights ($w$) and bias ($b$) are updated iteratively using gradient descent:

$$\frac{\partial C(w,b)}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$
$$\frac{\partial C(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})$$

and thus,

$${w} := {w} - \frac{\alpha}{m} * {X}^T(\hat{y} - y)$$
$$b := b - \frac{\alpha}{m} * \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})$$


Where:
- $\alpha$ is the learning rate.
- $X$ is the matrix of input features.
- $\hat{y}$ is the predicted output.


## Usage

To use the Logistic Regression model, follow these steps:

1. Import the `LogisticRegression` class from `logistic_regression.py`.
2. Create an instance of the `LogisticRegression` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
    from logistic_regression import LogisticRegression
    import numpy as np

    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis]

    logreg = LogisticRegression(learning_rate=0.1, num_iters=10000)
    logreg.fit(X, y)
    y_predict = logreg.predict(X)
