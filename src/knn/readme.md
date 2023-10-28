# K-Nearest Neighbor (KNN) Classification Algorithm

This Python script implements the K-nearest neighbor (KNN) classification algorithm from scratch, providing a basic understanding of the algorithm's inner workings.

## Math behind knn - todo

## Usage

To use the KNearestNeighbor model, follow these steps:

1. Import the `KNearestNeighbor` class from `knn.py`.
2. Create an instance of the `KNearestNeighbor` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
    X = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])

    KNN = KNearestNeighbor(k=1)
    KNN.fit(X, y)
    y_pred = KNN.predict(X, num_loops=2)
    accuracy = sum(y_pred == y) / y.shape[0]
    print(f"Accuracy: {accuracy}")
