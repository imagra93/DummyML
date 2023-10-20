"""
Implementation of K-nearest neighbor (KNN) from scratch
"""

import numpy as np


class KNearestNeighbor:
    def __init__(self, k):
        """
        Initialize K-nearest neighbor classifier with the specified value of k.

        Parameters:
        - k (int): Number of nearest neighbors to consider for classification.
        """
        self.k = k
        self.eps = 1e-8

    def fit(self, X, y):
        """
        Train the K-nearest neighbor classifier.

        Parameters:
        - X (numpy.ndarray): Training data features.
        - y (numpy.ndarray): Training data labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, num_loops=0):
        """
        Predict labels for test data using K-nearest neighbor classification.

        Parameters:
        - X_test (numpy.ndarray): Test data features.
        - num_loops (int): Number of loops to use in distance computation.
                           0: Vectorized, 1: One loop, 2: Two loops (naive).

        Returns:
        - numpy.ndarray: Predicted labels for the test data.
        """
        if num_loops == 0:
            distances = self.compute_distance_vectorized(X_test)

        elif num_loops == 1:
            distances = self.compute_distance_one_loop(X_test)

        else:
            distances = self.compute_distance_two_loops(X_test)

        return self.predict_labels(distances)

    def compute_distance_two_loops(self, X_test):
        """
        Inefficient naive implementation, use only
        as a way of understanding what kNN is doing.

        Parameters:
        - X_test (numpy.ndarray): Test data features.

        Returns:
        - numpy.ndarray: Pairwise distances between test and training data.
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(
                    self.eps + np.sum((X_test[i, :] - self.X_train[j, :]) ** 2)
                )

        return distances

    def compute_distance_one_loop(self, X_test):
        """
        Much better than two-loops but not as fast as fully vectorized version.
        Utilize Numpy broadcasting in X_train - X_test[i,:]

        Parameters:
        - X_test (numpy.ndarray): Test data features.

        Returns:
        - numpy.ndarray: Pairwise distances between test and training data.
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            distances[i, :] = np.sqrt(
                self.eps + np.sum((self.X_train - X_test[i, :]) ** 2, axis=1)
            )

        return distances

    def compute_distance_vectorized(self, X_test):
        """
        Can be tricky to understand this, we utilize heavy
        vecotorization as well as numpy broadcasting.
        Idea: if we have two vectors a, b (two examples)
        and for vectors we can compute (a-b)^2 = a^2 - 2a (dot) b + b^2
        expanding on this and doing so for every vector lends to the 
        heavy vectorized formula for all examples at the same time.

        Parameters:
        - X_test (numpy.ndarray): Test data features.

        Returns:
        - numpy.ndarray: Pairwise distances between test and training data.
        """
        X_test_squared = np.sum(X_test ** 2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train ** 2, axis=1, keepdims=True)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)

        return np.sqrt(
            self.eps + X_test_squared - 2 * two_X_test_X_train + X_train_squared.T
        )

    def predict_labels(self, distances):
        """
        Predict labels for test data based on K-nearest neighbors.

        Parameters:
        - distances (numpy.ndarray): Pairwise distances between test and training data.

        Returns:
        - numpy.ndarray: Predicted labels for the test data.
        """
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[: self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred


if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])

    KNN = KNearestNeighbor(k=1)
    KNN.fit(X, y)
    y_pred = KNN.predict(X, num_loops=2)
    accuracy = sum(y_pred == y) / y.shape[0]
    print(f"Accuracy: {accuracy}")
