"""
Implementation of Decision Tree model from scratch.
Metric used to apply the split on the data is the Gini index which is calculated for each feature's single value
in order to find the best split on each step. This means there is room for improvement performance wise as this
process is O(n^2) and can be reduced to linear complexity.

Parameters of the model:
max_depth (int): Maximum depth of the decision tree
min_node_size (int): Minimum number of instances a node can have. If this threshold is exceeded the node is terminated
"""

from collections import Counter
import numpy as np

class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        """
        Initialize the DecisionTree model.

        Parameters:
        - max_depth (int): Maximum depth of the decision tree.
        - min_node_size (int): Minimum number of instances a node can have. If this threshold is exceeded, the node is terminated.
        """
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}

    def calculate_gini(self, child_nodes):
        """
        Calculate the Gini index of a split in the dataset.

        Parameters:
        - child_nodes (list of np arrays): The groups of instances resulting from the split.

        Returns:
        - float: Gini index of the split.
        """
        n = sum(len(node) for node in child_nodes)
        gini = 0

        for node in child_nodes:
            m = len(node)
            if m == 0:
                continue

            y = [row[-1] for row in node]
            freq = Counter(y).values()

            node_gini = 1 - sum((i / m) ** 2 for i in freq)
            gini += (m / n) * node_gini

        return gini

    def apply_split(self, feature_index, threshold, data):
        """
        Split the dataset on a certain value of a feature.

        Parameters:
        - feature_index (int): Index of the selected feature.
        - threshold : Value of the feature split point.
        - data: Dataset.

        Returns:
        - np.array: Two new groups of split instances.
        """
        instances = data.tolist()
        left_child = np.array([row for row in instances if row[feature_index] < threshold])
        right_child = np.array([row for row in instances if row[feature_index] >= threshold])
        return left_child, right_child

    def find_best_split(self, data):
        """
        Find the best split on the dataset on each iteration of the algorithm by evaluating all possible splits and applying the one with the minimum Gini index.

        Parameters:
        - data: Dataset.

        Returns:
        - dict: Dictionary with the index of the splitting feature and its value and the two child nodes.
        """
        num_of_features = len(data[0]) - 1
        gini_score = float('inf')
        f_index = 0
        f_value = 0
        child_nodes = []

        for column in range(num_of_features):
            for row in data:
                value = row[column]
                l, r = self.apply_split(column, value, data)
                children = [l, r]
                score = self.calculate_gini(children)

                if score < gini_score:
                    gini_score = score
                    f_index, f_value, child_nodes = column, value, children

        return {"feature": f_index, "value": f_value, "children": child_nodes}

    def calc_class(self, node):
        """
        Calculate the most frequent class value in a group of instances.

        Parameters:
        - node: Group of instances.

        Returns:
        - Most common class value.
        """
        y = [row[-1] for row in node]
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]

    def recursive_split(self, node, depth):
        """
        Recursive function that builds the decision tree by applying split on every child node until they become terminal.

        Cases to terminate a node:
        i. max depth of tree is reached
        ii. minimum size of node is not met
        iii. child node is empty

        Parameters:
        - node: Group of instances.
        - depth (int): Current depth of the tree.
        """
        l, r = node["children"]
        del node["children"]

        if l.size == 0 or r.size == 0 or depth >= self.max_depth:
            c_value = self.calc_class(l) if l.size > 0 else self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return

        node["left"] = self.find_best_split(l)
        self.recursive_split(node["left"], depth + 1)

        node["right"] = self.find_best_split(r)
        self.recursive_split(node["right"], depth + 1)

    def train(self, X):
        """
        Apply the recursive split algorithm on the data in order to build the decision tree.

        Parameters:
        - X (np.array): Training data.

        Returns:
        - dict: The decision tree in the form of a dictionary.
        """
        tree = self.find_best_split(X)
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree

    def print_dt(self, tree, depth=0):
        """
        Print out the decision tree.

        Parameters:
        - tree (dict): Decision tree.
        - depth (int): Current depth of the tree.
        """
        if "feature" in tree:
            print("\nSPLIT NODE: feature #{} < {} depth:{}\n".format(tree["feature"], tree["value"], depth))
            self.print_dt(tree["left"], depth + 1)
            self.print_dt(tree["right"], depth + 1)
        else:
            print("TERMINAL NODE: class value:{} depth:{}".format(tree["class_value"], tree["depth"]))

    def predict_single(self, tree, instance):
        """
        Output the class value of the instance given based on the decision tree created previously.

        Parameters:
        - tree (dict): Decision tree.
        - instance (np.array): Single instance of data.

        Returns:
        - float: Predicted class value of the given instance.
        """
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1

        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    def predict(self, X):
        """
        Output the class value for each instance of the given dataset.

        Parameters:
        - X (np.array): Dataset with labels.

        Returns:
        - np.array: Array with the predicted class values of the dataset.
        """
        return np.array([self.predict_single(self.final_tree, row) for row in X])


if __name__ == "__main__":
    train_data = np.loadtxt("decisiontree/example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("decisiontree/example_data/targets.txt")

    dt = DecisionTree(max_depth=5, min_node_size=1)
    tree = dt.train(train_data)
    dt.print_dt(tree)
    y_pred = dt.predict(train_data)
    accuracy = sum(y_pred == train_y) / train_y.shape[0]
    print(f"Accuracy: {accuracy}")
