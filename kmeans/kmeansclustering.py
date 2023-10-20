import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansClustering:
    """
    Implementation of K-means clustering algorithm.

    Parameters:
    - num_clusters (int): Number of clusters.
    """
    def __init__(self, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.plot_figure = True
        self.centroids = None  # Centroids will be saved within the object

    def initialize_random_centroids(self, X):
        """
        Initialize centroids by randomly selecting data points.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Initial centroids.
        """
        centroids = np.zeros((self.K, X.shape[1]))

        for k in range(self.K):
            centroid = X[np.random.choice(range(X.shape[0]))]
            centroids[k] = centroid

        return centroids

    def create_clusters(self, X):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - list: List of clusters, where each cluster contains indices of data points.
        """
        clusters = [[] for _ in range(self.K)]

        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - self.centroids) ** 2, axis=1))
            )
            clusters[closest_centroid].append(point_idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        """
        Update centroids based on the mean of data points in each cluster.

        Parameters:
        - clusters (list): List of clusters.
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Updated centroids.
        """
        centroids = np.zeros((self.K, X.shape[1]))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids

    def fit(self, X):
        """
        Fit the K-means clustering algorithm to the data.

        Parameters:
        - X (numpy.ndarray): Input data.
        """
        self.centroids = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X)

            previous_centroids = self.centroids
            self.centroids = self.calculate_new_centroids(clusters, X)

            diff = self.centroids - previous_centroids

            if not diff.any():
                print("Termination criterion satisfied")
                break

        if self.plot_figure:
            self.plot_fig(X, self.predict(X))

    def predict(self, X):
        """
        Predict cluster labels for input data.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Predicted cluster labels.
        """
        clusters = self.create_clusters(X)
        y_pred = np.zeros(X.shape[0])

        for cluster_idx, cluster in enumerate(clusters):
            y_pred[cluster] = cluster_idx

        return y_pred

    def plot_fig(self, X, y, filename='kmeans_plot.png'):
        """
        Save the plot of input data points with cluster assignments.

        Parameters:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): Predicted cluster labels.
        - filename (str): Name of the file to save the plot.
        """
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.savefig(filename)

if __name__ == "__main__":
    np.random.seed(10)
    num_clusters = 3
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)

    Kmeans = KMeansClustering(num_clusters)
    Kmeans.fit(X)
