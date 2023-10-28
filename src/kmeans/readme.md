# K-Means Clustering Algorithm

This Python script implements the K-means clustering algorithm, a popular unsupervised machine learning technique used for partitioning a dataset into K distinct, non-overlapping subsets (clusters).

## Algorithm Overview

The K-means algorithm consists of the following steps:

- Initialization:
    Randomly select K data points as initial centroids.

- Assignment:
    Assign each data point to the nearest centroid.

- Update:
    Update the centroids based on the mean of data points in each cluster.

- Repeat:
    Repeat the assignment and update steps until convergence or a maximum number of iterations.


## Usage

To use the KMeansClustering model, follow these steps:

1. Import the `KMeansClustering` class from `kmeans.py`.
2. Create an instance of the `KMeansClustering` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kmeans_clustering import KMeansClustering

np.random.seed(10)
num_clusters = 3
X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)

Kmeans = KMeansClustering(num_clusters)

Kmeans.fit(X)
