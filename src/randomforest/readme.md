# Random Forest Implementation
This Python code implements a Random Forest classifier from scratch. The Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) of the individual trees.

## Parameters
- n_trees: Number of decision trees in the Random Forest.
- fold_size: Percentage of the dataset used for each tree during training.
- feat_fold_size: Percentage of the dataset's features used for each tree during training.
- max_depth: Maximum depth of each decision tree.
- min_node_size: Minimum number of instances a node can have, terminating the node if exceeded.
- criterion: The splitting criterion for decision tree nodes, either "gini" or "entropy".


## Usage

To use the RandomForest model, follow these steps:

1. Import the `RandomForest` class from `random_forest.py`.
2. Create an instance of the `RandomForest` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    rf = RandomForest(n_trees=100, fold_size=0.8, criterion='gini', max_depth=5, feat_fold_size=0.7)
    rf.fit(X, y)
    y_pred = rf.predict(X)
