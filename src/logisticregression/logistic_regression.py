"""
Implementation of Logistic Regression using Gradient Descent.

Let
    m = #training examples,
    n = #number of features

It takes as input the following:
    X is R^(m x n),
    y is R^(m x 1),
    weights is R^(n x 1),
    bias is a scalar.
"""

import numpy as np
from sklearn.datasets import make_blobs

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iters=1000, print_cost=True):
        """
        Initialize the logistic regression model.

        Parameters:
        - learning_rate: The learning rate for gradient descent.
        - num_iters: The number of iterations for gradient descent.
        """
        self.lr = learning_rate
        self.num_iters = num_iters
        self.weights = None
        self.bias = None
        self.cost_list = None
        self.print_cost = print_cost

    def sigmoid(self, z):
        """
        Calculate the sigmoid function.

        Parameters:
        - z: Input value.

        Returns:
        - Sigmoid of the input value.
        """
        return 1 / (1 + np.exp(-z))

    def cost(self, y_predict, y):
        """
        Calculate the logistic loss/cost.

        Parameters:
        - y_predict: Predicted probabilities.
        - y: True labels.

        Returns:
        - Logistic loss/cost.
        """
        return -1 / self.m * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))

    def train(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        - X: Input features.
        - y: True labels.

        Returns:
        - Trained weights and bias.
        """
        self.m, self.n = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((self.n, 1))  # Could be random (convex func.)
        self.bias = 0

        if self.print_cost: 
            print(f"Training with gradient descent ({self.num_iters} epochs)")

        self.cost_list = []

        for it in range(self.num_iters):
            # Calculate hypothesis
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Calculate cost
            cost = self.cost(y_predict, y)
            self.cost_list.append(cost)

            # Backpropagation / Gradient calculations
            dw = 1 / self.m * np.dot(X.T, (y_predict - y))
            db = 1 / self.m * np.sum(y_predict - y)

            # Gradient descent update step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print cost sometimes
            if it % 100 == 0 and self.print_cost:
                print(f"   -Cost after iteration {it}: {cost:.5f}")

        return self.weights, self.bias

    def predict(self, X, threshold=0.5):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Input features.
        - threshold: Threshold for binary classification.

        Returns:
        - Predicted labels.
        """
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_labels = y_predict > threshold
        return y_predict_labels.astype(int)

    def calculate_accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the predictions.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels.

        Returns:
        - Accuracy.
        """
        return np.sum(y_true.flatten() == y_pred.flatten()) / len(y_true)

    def fit(self, X, y):
        """
        Fit the logistic regression model.

        Parameters:
        - X: Input features.
        - y: True labels.

        Returns:
        - Trained weights and bias.
        """
        return self.train(X, y)

    def fit_predict(self, X, y):
        """
        Fit the model and make predictions.

        Parameters:
        - X: Input features.
        - y: True labels.

        Returns:
        - Predicted labels.
        """
        self.fit(X, y)
        y_predict = self.predict(X)
        return y_predict

if __name__ == "__main__":
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis]

    logreg = LogisticRegression(learning_rate=0.1, num_iters=10000)
    logreg.fit(X, y)
    y_predict = logreg.predict(X)

    accuracy = logreg.calculate_accuracy(y, y_predict)
    print(f"Accuracy: {accuracy}")
