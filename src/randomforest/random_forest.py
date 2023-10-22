import math
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)
from src.decisiontree.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=20, fold_size=0.7, criterion='gini', max_depth=5, min_node_size=1, feat_fold_size=0.7):
        """
        Initialize Random Forest model.

        Parameters:
        - n_trees (int): The total number of trees in the forest.
        - fold_size (int): Percentage of the original dataset size each fold should be.
        - max_depth (int): Maximum depth of the decision tree.
        - min_node_size (int): Minimum number of instances a node can have. If this threshold is exceeded, the node is terminated.
        - criterion (str): Splitting criterion, either "gini" or "entropy".
        """
        self.n_trees = n_trees
        self.fold_size = fold_size
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.feat_fold_size = feat_fold_size
        self.trees = list()
        self.trees_features = list()

    def cross_validation_split(self, dataset, n_folds, p):
        """
        Split the given dataset into n-folds with replacement.

        Parameters:
        - dataset (np.array): The input dataset.
        - n_folds (int): Number of folds in which the dataset should be split.
        - p (int): Percentage of the dataset's size the size of a single fold should be.

        Returns:
        - list of np arrays: List with the k-folds.
        """
        dataset_split = list()
        fold_size = min(math.ceil(len(dataset) * p), len(dataset))
        print(f'  -Each tree will use {fold_size} out of {len(dataset)} samples')
        for _ in range(n_folds):
            # select fold_size random samples.
            selected_samples = np.random.choice(len(dataset), size=fold_size, replace=False)
            dataset_split.append(dataset[selected_samples, :])
        return dataset_split

    def randomize_features(self, splits):
        """
        Randomize the selection of features for each tree.

        Parameters:
        - splits (list of np arrays): List of folds.

        Returns:
        - list of np arrays: List with the k-folds with some features randomly removed.
        """
        dataset_split = list()
        l = len(splits[0][0]) - 1
        n_features = min(math.ceil(l * self.feat_fold_size), l)  # random proportion of features
        print(f'  -Each tree will use {n_features} out of {l} features')
        for split in splits:
            # select n_features random features.
            selected_features = np.random.choice(l, size=n_features, replace=False)
            # Append l to the end (label column)
            selected_features = np.append(selected_features, l)
            dataset_split.append(split[:, selected_features])
            # Save selected features for each tree (needed in predict)
            self.trees_features.append(selected_features[:-1])
        return dataset_split

    def fit(self, X, y):
        """
        Train the Random Forest model.

        Parameters:
        - X (np.array): Training data.
        - y (np.array): Target labels.
        """
        print(f'Fitting random forest with {self.n_trees} trees')
        # We stack the classes y at the end of the features X.
        train_x = self.cross_validation_split(np.column_stack((X, y)), self.n_trees, self.fold_size)
        train_x = self.randomize_features(train_x)
        for fold, features_used in zip(train_x, self.trees_features):
            dt = DecisionTree(self.max_depth, self.min_node_size, criterion=self.criterion)
            dt.fit(fold[:, :-1], fold[:, -1])  # Separate features and labels
            self.trees.append(dt)

    def predict(self, X, per_tree=False):
        """
        Predict the class value for each instance of the given dataset using the random forest algorithm.

        Parameters:
        - X (np.array): Dataset with labels.
        - per_tree (bool): return output of each individual tree.

        Returns:
        - tuple: Array with the predicted class values of the dataset and individual predictions from each tree.
        """
        predicts = list()
        final_predicts = list()
        for tree, features_used in zip(self.trees, self.trees_features):
            # Use only the features used during training for each tree
            X_subset = X[:, features_used]
            predicts.append(tree.predict(X_subset))
        # iterate through each tree's class prediction and find the most frequent for each instance
        for i in range(len(predicts[0])):
            values = list()
            for j in range(len(predicts)):
                values.append(predicts[j][i])
            final_predicts.append(max(set(values), key=values.count))
        if per_tree:
            return final_predicts, predicts
        else:
            return final_predicts


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and train Random Forest model
    rf = RandomForest(n_trees=100, fold_size=0.8, criterion='gini', max_depth=5, feat_fold_size=0.7)
    rf.fit(X_train, y_train)

    # Evaluate Random Forest on training data
    y_pred_rf_train, y_pred_ind_rf_train = rf.predict(X_train, per_tree=True)
    accuracy_rf_train = sum(y_train == y_pred_rf_train) / len(y_pred_rf_train)
    print(f"Accuracy of Random Forest on training data: {accuracy_rf_train:.3f}")

    # Print accuracy for each individual tree
    print("Accuracy for each individual tree on training data:")
    print([round(sum(y_train == pred) / len(pred), 2) for pred in y_pred_ind_rf_train])

    # Evaluate Random Forest on testing data
    y_pred_rf, y_pred_ind_rf = rf.predict(X_test, per_tree=True)
    accuracy_rf = sum(y_test == y_pred_rf) / len(y_pred_rf)
    print(f"\nAccuracy of Random Forest on testing data: {accuracy_rf:.3f}")

    # Print accuracy for each individual tree
    print("Accuracy for each individual tree on testing data:")
    print([round(sum(y_test == pred) / len(pred),2) for pred in y_pred_ind_rf])

    # Build and train a single Decision Tree
    single_tree = DecisionTree(criterion='gini')
    single_tree.fit(X_train, y_train)

    # Evaluate Decision Tree on testing data
    y_pred_dt_train = single_tree.predict(X_train)
    accuracy_dt = sum(y_train == y_pred_dt_train) / len(y_pred_dt_train)
    print(f"\nAccuracy of single Decision Tree on training data: {accuracy_dt:.3f}")

    # Evaluate Decision Tree on testing data
    y_pred_dt = single_tree.predict(X_test)
    accuracy_dt = sum(y_test == y_pred_dt) / len(y_pred_dt)
    print(f"Accuracy of single Decision Tree on testing data: {accuracy_dt:.3f}")
