"""
Naive Bayes Classifier Implementation from scratch

To run the code structure the code in the following way:
    X be size: (num_training_examples, num_features)
    y be size: (num_classes, )

Where the classes are 0, 1, 2, etc. Then an example run looks like:
    NB = NaiveBayes(X, y)
    NB.fit(X)
    predictions = NB.predict(X)
"""
import numpy as np

class NaiveBayes:
    def __init__(self, X, y):
        """Initialize the NaiveBayes model with data dimensions."""
        self.num_examples, self.num_features = X.shape
        self.classes = np.unique(y)
        self.eps = 1e-6

    def fit(self, X):
        """Fit the NaiveBayes model to the training data."""
        # Initialize dictionaries to store mean, variance, and prior for each class
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        # Calculate mean, variance, and prior for each class
        for c in self.classes:
            X_c = X[y == c]

            self.classes_mean[c] = np.mean(X_c, axis=0)
            self.classes_variance[c] = np.var(X_c, axis=0)
            self.classes_prior[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        """Make predictions using the trained NaiveBayes model."""
        # Initialize an array to store class probabilities for each example
        probs = np.zeros((self.num_examples, len(self.classes)))

        # Calculate log probabilities for each class
        for index, c in enumerate(self.classes):
            prior = self.classes_prior[c]
            probs_c = self.density_function(
                X, 
                self.classes_mean[c], 
                self.classes_variance[c]
            )
            probs[:, index] = probs_c + np.log(prior)

        # Predict the class with the highest probability
        return self.classes[np.argmax(probs, 1)]

    def density_function(self, x, mean, sigma):
        """
        Calculate probability using the log Gaussian density function.

        Parameters:
        - x (numpy.ndarray): Input data.
        - mean (numpy.ndarray): Mean of the Gaussian distribution.
        - sigma (numpy.ndarray): Standard deviation of the Gaussian distribution.

        Returns:
        numpy.ndarray: Array of probabilities for each data point.

        Note: The implementation assumes a diagonal covariance matrix.

        """
        # The log of the Gaussian density function part (2*pi)^(-k/2)*prod(sigma)^(-1/2)
        const = - self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps)) # constant for all (could be avoided)
        # The log of exponential part exp(1/2*(x-mu)^2)
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1) 
        return const - probs


# Example usage
if __name__ == "__main__":
    # Load example data
    X = np.loadtxt("naivebayes/example_data/data.txt", delimiter=",")
    y = np.loadtxt("naivebayes/example_data/targets.txt") - 1

    # Create NaiveBayes object, fit the model, and make predictions
    NB = NaiveBayes(X, y)
    NB.fit(X)
    y_pred = NB.predict(X)

    # Print accuracy
    print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")
