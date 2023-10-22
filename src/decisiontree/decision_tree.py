from collections import Counter
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, min_node_size=1, criterion="entropy"):
        """
        Initialize the DecisionTree model.

        Parameters:
        - max_depth (int): Maximum depth of the decision tree.
        - min_node_size (int): Minimum number of instances a node can have. If this threshold is exceeded, the node is terminated.
        - criterion (str): Splitting criterion, either "gini" or "entropy".
        """
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.tree = None
        self.criterion = criterion

    def calculate_criterion(self, child_nodes):
        """
        Calculate the specified criterion (Gini or Entropy) for a set of child nodes.

        Parameters:
        - child_nodes (list of np arrays): The groups of instances resulting from the split.

        Returns:
        - float: Gini index or Entropy of the split.
        """
        if self.criterion == "gini":
            return self.calculate_gini(child_nodes)
        elif self.criterion == "entropy":
            return self.calculate_entropy(child_nodes)
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'.")

    def calculate_entropy(self, child_nodes):
        """
        Calculate the Entropy of a split in the dataset.

        Parameters:
        - child_nodes (list of np arrays): The groups of instances resulting from the split.

        Returns:
        - float: Entropy of the split.
        """
        n = sum(len(node) for node in child_nodes)
        entropy = 0

        for node in child_nodes:
            m = len(node)
            if m == 0:
                continue

            y = [row[-1] for row in node]
            freq = Counter(y).values()

            # Entropy Formula: H(S) = -sum(p_i * log2(p_i))
            node_entropy = -sum((i / m) * np.log2(i / m) if i > 0 else 0 for i in freq)
            entropy += (m / n) * node_entropy

        return entropy

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

            y = [row[-1] for row in node] # Last value is the class.
            freq = Counter(y).values()

            # Gini Index Formula: Gini(S) = 1 - sum(p_i^2)
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
        Find the best split on the dataset based on the specified criterion.

        Parameters:
        - data: Dataset.

        Returns:
        - dict: Dictionary with the index of the splitting feature and its value and the two child nodes.
        """
        num_of_features = len(data[0]) - 1 # Last value not a feature.
        criterion_score = float('inf')
        f_index = 0
        f_value = 0
        child_nodes = []

        for feat_idx in range(num_of_features):
            for row in data:
                value = row[feat_idx]
                l, r = self.apply_split(feat_idx, value, data)
                children = [l, r]
                score = self.calculate_criterion(children)

                if score < criterion_score:
                    criterion_score = score
                    f_index, f_value, child_nodes = feat_idx, value, children

        return {"feature": f_index, "value": f_value, "children": child_nodes}

    def calc_class(self, node):
        """
        Calculate the most frequent class value in a group of instances.

        Parameters:
        - node: Group of instances.

        Returns:
        - Most common class value.
        """
        y = [row[-1] for row in node] # Last value is the class.
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

    def fit(self, X, y):
        """
        Apply the recursive split algorithm on the data in order to build the decision tree.

        Parameters:
        - X (np.array): Training data.
        - y (np.array): Target labels.
        """
        data = np.column_stack((X, y))  # Combine features and labels into one array
        self.tree = self.find_best_split(data)
        self.recursive_split(self.tree, 1)

    def predict_single(self, instance):
        """
        Output the class value of the instance given based on the decision tree created previously.

        Parameters:
        - instance (np.array): Single instance of data.

        Returns:
        - float: Predicted class value of the given instance.
        """
        if self.tree is None:
            print("ERROR: Please train the decision tree first")
            return -1

        tree = self.tree
        while "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                tree = tree["left"]
            else:
                tree = tree["right"]

        return tree["class_value"]

    def predict(self, X):
        """
        Output the class value for each instance of the given dataset.

        Parameters:
        - X (np.array): Dataset with labels.

        Returns:
        - np.array: Array with the predicted class values of the dataset.
        """
        return np.array([self.predict_single(row) for row in X])


if __name__ == "__main__":
    train_data = np.loadtxt("src/decisiontree/example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("src/decisiontree/example_data/targets.txt")

    dt = DecisionTree(max_depth=4, min_node_size=1, criterion="entropy") # gini or entropy
    dt.fit(train_data, train_y)
    y_pred = dt.predict(train_data)
    accuracy = sum(y_pred == train_y) / train_y.shape[0]
    print(f"Accuracy: {accuracy:.3f}")
