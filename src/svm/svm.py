import numpy as np
import cvxopt
from utils import create_dataset, plot_contour

class SVM:
    def __init__(self, kernel='linear', C=1, degree=5, sigma=0.1):
        """
        Support Vector Machine (SVM) classifier.

        Parameters:
        - kernel_type (str): Type of kernel - 'linear', 'polynomial', or 'gaussian'.
        - C (float): Regularization parameter.
        - degree (int): Degree of the polynomial kernel (only applicable for polynomial kernel).
        - sigma (float): Sigma value for the Gaussian kernel (only applicable for Gaussian kernel).
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.sigma = sigma

    def linear_kernel(self, x, z):
        """Linear kernel function."""
        return np.dot(x, z.T)

    def polynomial_kernel(self, x, z):
        """Polynomial kernel function."""
        return (1 + np.dot(x, z.T)) ** self.degree

    def gaussian_kernel(self, x, z):
        """Gaussian (RBF) kernel function."""
        return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (self.sigma ** 2)))

    def select_kernel(self):
        """Select the appropriate kernel based on the specified type."""
        if self.kernel == 'linear':
            return self.linear_kernel
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel
        elif self.kernel == 'gaussian':
            return self.gaussian_kernel
        else:
            raise ValueError("Invalid kernel type. Supported types: 'linear', 'polynomial', 'gaussian'")
        
    def optimize_svm_dual_form(self, K, y, C):
        """
        Optimize the dual form of the Support Vector Machine (SVM) problem using the cvxopt library.

        This function solves the following quadratic programming problem to obtain the Lagrange multipliers (alphas)
        for the SVM:

            minimize    (1/2) * alphas^T * P * alphas + q^T * alphas
            subject to  G * alphas <= h
                        A * alphas = b

        where:
        - P is a positive semidefinite matrix defined as np.outer(y, y) * K
        - q is a column vector defined as -np.ones((m, 1))
        - G is a matrix stacking two block matrices: np.eye(m) * -1 and np.eye(m)
        - h is a column vector stacking two block vectors: np.zeros(m) and np.ones(m) * C
        - A is a row vector defined as y reshaped to a 1x(m) matrix
        - b is a scalar with value 0
        - K is the kernel matrix
        - C is the regularization parameter
        - y is the vector of class labels
        - m is the number of training examples

        The solution is obtained using the quadratic programming solver from the cvxopt library.

        Parameters:
            K (numpy.ndarray): Kernel matrix.
            y (numpy.ndarray): Vector of class labels.
            C (float): Regularization parameter.

        Returns:
            numpy.ndarray: Optimized Lagrange multipliers (alphas).
        """
        m = len(y)

        # Construct the quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.zeros(1))

        # Solve the quadratic programming problem
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extract and return the optimized alphas
        alphas = np.array(sol["x"])
        return alphas

    def fit(self, X, y):
        """
        Train the SVM classifier.

        Parameters:
        - X (numpy array): Training data.
        - y (numpy array): Training labels.
        """
        self.y = y
        self.X = X
        m, _ = X.shape

        # Select the appropriate kernel function
        self.kernel_function = self.select_kernel()

        # Calculate Kernel
        K = np.zeros((m, m))
        for i in range(m):
            K[i, :] = self.kernel_function(X[i, np.newaxis], self.X)

        # Solve with cvxopt
        self.alphas = self.optimize_svm_dual_form(K, y, self.C)
       
        # Once alphas are obtained.
        sv_boolean = self.get_support_vectors_booleans()
        self.b = np.mean(
            self.y[sv_boolean, np.newaxis]
            - self.alphas[sv_boolean] * self.y[sv_boolean, np.newaxis] * K[sv_boolean, sv_boolean][:, np.newaxis]
        )
        self.support_vectors_ = self.X[sv_boolean]

    def predict(self, X):
        """
        Predict labels for new data points.

        Parameters:
        - X (numpy array): Data points to predict.

        Returns:
        - numpy array: Predicted labels.
        """
        y_predict = np.zeros((X.shape[0]))
        sv_boolean = self.get_support_vectors_booleans()

        for i in range(X.shape[0]):
            y_predict[i] = np.sum(
                self.alphas[sv_boolean]
                * self.y[sv_boolean, np.newaxis]
                * self.kernel_function(X[i], self.X[sv_boolean])[:, np.newaxis]
            ) + self.b

        if self.kernel == 'linear':
            """
            This comes from the >1 decision boundary but,
            is not working with poly or gaussian.
            If I do not put it in the linear, the decission boundary
            is not at the middle of the groups
            """
            
            y_predict += 1

        return np.sign(y_predict)

    def get_support_vectors_booleans(self):
        """
        Identify support vectors.

        Returns:
        - numpy array: Boolean mask indicating support vectors.
        """
        threshold = 1e-6
        sv_boolean = (self.alphas > threshold).flatten()
        return sv_boolean

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

    # Linear classifier
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=50, centers=2,
                    random_state=0, cluster_std=0.6)
    y[y==0] = -1

    svm = SVM(C=1e4, kernel='linear')
    y_pred = svm.fit_predict(X, y)
    plot_contour(X, y, svm, title='linear')

    accuracy = sum(y == y_pred) / y.shape[0]
    print(f"Accuracy: {accuracy}")

    print("Support vectors (linear case):")
    print(svm.support_vectors_)

    # Gaussian classifier
    np.random.seed(1)
    X, y = create_dataset(N=50)
    
    svm = SVM(C=1e4, kernel='gaussian')
    y_pred = svm.fit_predict(X, y)
    plot_contour(X, y, svm, title='gaussian')

    accuracy = sum(y == y_pred) / y.shape[0]
    print(f"Accuracy: {accuracy}")

    # Example usage with polynomial kernel
    svm = SVM(C=1e4, kernel='polynomial', degree=3)
    y_pred = svm.fit_predict(X, y)
    plot_contour(X, y, svm, title='poly')

    accuracy = sum(y == y_pred) / y.shape[0]
    print(f"Accuracy: {accuracy}")
