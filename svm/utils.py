import numpy as np
import matplotlib.pyplot as plt

def create_dataset(N, D=2, K=2):
    """
    Generate a synthetic dataset for testing classifiers.

    Parameters:
    - N (int): Number of points per class.
    - D (int): Number of features.
    - K (int): Number of classes.

    Returns:
    - X (numpy array): Data matrix (each row represents a single example).
    - y (numpy array): Class labels.
    """
    X = np.zeros((N * K, D))  # Data matrix (each row = single example)
    y = np.zeros(N * K)  # Class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # Radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # Theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.savefig('dataset_plot.png')
    plt.show()

    y[y == 0] -= 1

    return X, y

def plot_contour(X, y, svm):
    """
    Plot decision boundaries and data points for a given SVM classifier.

    Parameters:
    - X (numpy array): Data matrix.
    - y (numpy array): Class labels.
    - svm (SVM classifier): Trained SVM classifier.
    """
    # Plot the resulting classifier
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Plot the points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.savefig('svm_decision_boundary.png')
    plt.show()
