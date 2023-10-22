import numpy as np
import cvxopt
from utils import create_dataset, plot_contour

def linear_kernel(x, z):
    """Linear kernel function."""
    return np.dot(x, z.T)

def polynomial_kernel(x, z, p=5):
    """Polynomial kernel function."""
    return (1 + np.dot(x, z.T)) ** p

def gaussian_kernel(x, z, sigma=0.1):
    """Gaussian (RBF) kernel function."""
    return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (sigma ** 2)))

class SVM:
    def __init__(self, kernel=gaussian_kernel, C=1):
        """
        Support Vector Machine (SVM) classifier.

        Parameters:
        - kernel (function): Kernel function.
        - C (float): Regularization parameter.
        """
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        """
        Train the SVM classifier.

        Parameters:
        - X (numpy array): Training data.
        - y (numpy array): Training labels.
        """
        self.y = y
        self.X = X
        m, n = X.shape

        # Calculate Kernel
        self.K = np.zeros((m, m))
        for i in range(m):
            self.K[i, :] = self.kernel(X[i, np.newaxis], self.X)

        # Solve with cvxopt
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])

    def predict(self, X):
        """
        Predict labels for new data points.

        Parameters:
        - X (numpy array): Data points to predict.

        Returns:
        - numpy array: Predicted labels.
        """
        y_predict = np.zeros((X.shape[0]))
        support_vectors = self.get_support_vectors()

        for i in range(X.shape[0]):
            y_predict[i] = np.sum(
                self.alphas[support_vectors]
                * self.y[support_vectors, np.newaxis]
                * self.kernel(X[i], self.X[support_vectors])[:, np.newaxis]
            )

        return np.sign(y_predict + self.b)

    def get_support_vectors(self):
        """
        Identify support vectors.

        Returns:
        - numpy array: Boolean mask indicating support vectors.
        """
        threshold = 1e-5
        sv = ((self.alphas > threshold) * (self.alphas < self.C)).flatten()
        self.b = np.mean(
            self.y[sv, np.newaxis]
            - self.alphas[sv] * self.y[sv, np.newaxis] * self.K[sv, sv][:, np.newaxis]
        )
        return sv

    def fit_predict(self, X, y):
        """
        Train the SVM classifier and predict labels for new data points.

        Parameters:
        - X (numpy array): Training data.
        - y (numpy array): Training labels.

        Returns:
        - numpy array: Predicted labels.
        """
        self.fit(X, y)
        return self.predict(X)

if __name__ == "__main__":
    np.random.seed(1)
    X, y = create_dataset(N=50)

    svm = SVM(kernel=polynomial_kernel)
    y_pred = svm.fit_predict(X, y)
    plot_contour(X, y, svm)

    accuracy = sum(y == y_pred) / y.shape[0]
    print(f"Accuracy: {accuracy}")
