# Naive Bayes Classifier

This repository contains a Python implementation of the Naive Bayes Classifier from scratch. The Naive Bayes Classifier is a probabilistic machine learning algorithm based on Bayes' theorem. It is particularly useful for classification tasks, such as spam filtering and document categorization.

## math behind naive bayes

The Naive Bayes algorithm is based on Bayes' Theorem, which relates the conditional and marginal probabilities of random events. Here are the formulas for a basic Gaussian Naive Bayes classifier:

1. Prior Probability $P(C_i)$:
   This represents the prior probability of class C_i, the probability of encountering class C_i without considering any features.

2. Likelihood:
   $$ P(X | C_i) = \prod_{j=1}^{n} \frac{1}{\sqrt{(2pi*\sigma_{ij}^2)}} * \exp(\frac{-(x_j - mu_{ij})^2}{(2*\sigma_{ij}^2)})$$
   - This represents the likelihood of the features $X$ given class $C_i$. It's modeled using the Gaussian (normal) distribution.

3. Posterior Probability:
   $$P(C_i | X) = P(X | C_i) * P(C_i) / P(X)$$
   This is the probability of class $C_i$ given the observed features $X$, calculated using Bayes' Theorem. The denominator $P(X)$ is often ignored for comparison.

   Log-Likelihood:
   $$log\_likelihood = \sum_{j=1}^{n} (-\frac{1}{2} * log(2*\pi) - \frac{1}{2} * log(\sigma_{ij}^2) - \frac{(x_j - \mu_{ij})^2}{2*\sigma_{ij}^2}$$
   
   which is used for numerical stability.

## Usage

To use the NaiveBayes model, follow these steps:

1. Import the `NaiveBayes` class from `naivebayes.py`.
2. Create an instance of the `NaiveBayes` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
    X = np.loadtxt("src/naivebayes/example_data/data.txt", delimiter=",")
    y = np.loadtxt("src/naivebayes/example_data/targets.txt") - 1

    # Create NaiveBayes object, fit the model, and make predictions
    NB = NaiveBayes(X, y)
    NB.fit(X)
    y_pred = NB.predict(X)