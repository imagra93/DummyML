"""
Implementation of Linear Regression using Gradient Descent.

Let
    m = #training examples,
    n = #number of features Sizes differ 

It takes as input the following:
    y is R^(1 x m),
    X is R^(n x m), 
    w is R^(n x 1)
"""

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, total_iterations=1000, print_cost=True):
        """
        Initialize the linear regression model.

        Parameters:
        - learning_rate: The learning rate for gradient descent.
        - total_iterations: The number of iterations for gradient descent.
        - print_cost: Whether to print the cost during training.
        """
        self.learning_rate = learning_rate
        self.num_iters = total_iterations
        self.print_cost = print_cost
        self.weights = None

    def y_hat(self, X):
        """Predicted output based on current weights."""
        return np.dot(self.weights.T, X)

    def cost(self, yhat, y):
        """Calculate mean squared error cost."""
        C = 1 / self.m * np.sum(np.power(yhat - y, 2))
        return C

    def gradient_descent(self, X, y, yhat):
        """Update weights using gradient descent."""
        dCdW = 2 / self.m * np.dot(X, (yhat - y).T)
        self.weights = self.weights - self.learning_rate * dCdW

    def mean_squared_error(self, y_true, y_pred):
        """Calculate Mean Squared Error (MSE)."""
        return np.mean(np.square(y_true - y_pred))

    def fit(self, X, y):
        """Fit the linear regression model with gd."""
        # Add x0 = 1 to represent the intercept term
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)

        self.m = X.shape[1]
        self.n = X.shape[0]

        self.weights = np.zeros((self.n, 1))  # Could be random (convex func.)

        if self.print_cost: 
            print(f"Training with gradient descent ({self.num_iters} epochs)")

        for it in range(self.num_iters):
            yhat = self.y_hat(X)
            cost = self.cost(yhat, y)

            if it % 100 == 0 and self.print_cost:
                print(f"   -Iteration {it}: Cost = {cost:.5f}")

            self.gradient_descent(X, y, yhat)

    def predict(self, X):
        """Make predictions based on the learned weights."""
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)
        return self.y_hat(X)

    def fit_predict(self, X, y):
        """Fit the model and make predictions."""
        self.fit(X, y)
        return self.predict(X)

    def fit_normal(self, X, y):
        """Fit the linear regression model with normal eq."""
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)
        self.weights = np.dot(np.linalg.pinv(np.dot(X, X.T)), np.dot(X, y.T))

if __name__ == "__main__":
    # Generate random data for testing
    X = np.random.rand(1, 500)
    y = 3 * X + 5 + np.random.randn(1, 500) * 0.1

    # Initialize and run linear regression
    regression = LinearRegression(learning_rate=0.01, total_iterations=1000)
    predictions = regression.fit_predict(X, y)

    # Calculate and display MSE
    mse = regression.mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse}")

    # Display the learned weights and predictions
    print("Learned Weights:")
    print(regression.weights)

    # Normal equation
    regression.fit_normal(X, y)
    print("Learned Weights (normal eq.):")
    print(regression.weights)

    import matplotlib.pyplot as plt

    # Plot the data and the final predicted line
    plt.scatter(X, y, label='Data')
    plt.scatter(X, predictions, color='red', label='Predicted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression with Gradient Descent')
    plt.savefig('linear_regression_plot.png')

    print("Testing with multiple features")
    # Generate random data for testing
    X = np.random.rand(2, 500)
    y = 3 * X[0,:] + X[1,:] + 5 + np.random.randn(1, 500) * 0.1

    # Initialize and run linear regression
    regression = LinearRegression(learning_rate=0.01, total_iterations=5000)
    predictions = regression.fit_predict(X, y)

    # Calculate and display MSE
    mse = regression.mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse}")

    # Display the learned weights and predictions
    print("Learned Weights:")
    print(regression.weights)

    # Normal equation
    regression.fit_normal(X, y)
    print("Learned Weights (normal eq.):")
    print(regression.weights)
