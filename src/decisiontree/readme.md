# DecisionTree

The DecisionTree class is an implementation of a decision tree algorithm for classification tasks. Decision trees are a type of supervised machine learning algorithm that partitions the data into subsets based on features to make predictions.

## Parameters
- max_depth (int): Maximum depth of the decision tree.
- min_node_size (int): Minimum number of instances a node can have. Nodes with fewer instances are terminated.
- criterion (str): Splitting criterion, either "gini" or "entropy."

## Math Behind the DecisionTree

### Entropy Calculation

Entropy measures the impurity or disorder in a set. For a split in the dataset, the entropy is calculated using the formula:

$$ H(S) = -\sum_{i} p_i \log_2(p_i) $$

where $p_i$ is the proportion of instances of class $i$ in a node.

### Gini Index Calculation

Gini index measures the impurity of a set, with a lower value indicating a purer set. The Gini index for a split is calculated as:

$$ Gini(S) = 1 - \sum_{i} (p_i)^2 $$

where $p_i$ is the proportion of instances of class $i$ in a node.

## Recursive Splitting

The decision tree is built by recursively applying optimal splits (the ones that maximize information gain) on child nodes until they become terminal. Nodes are terminated based on:

- Maximum depth of the tree is reached.
- Minimum size of a node is not met.
- Child node is empty.


## Usage

To use the DecisionTree model, follow these steps:

1. Import the `DecisionTree` class from `decision_tree.py`.
2. Create an instance of the `DecisionTree` class with optional parameters.
3. Call the `fit` method with training data (X, y) to train the model.
4. Use the `predict` method to make predictions on new data.

```python
    train_data = np.loadtxt("src/decisiontree/example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("src/decisiontree/example_data/targets.txt")

    dt = DecisionTree(max_depth=4, min_node_size=1, criterion="entropy") # gini or entropy
    dt.fit(train_data, train_y)
    y_pred = dt.predict(train_data)
    accuracy = sum(y_pred == train_y) / train_y.shape[0]
    print(f"Accuracy: {accuracy:.3f}")
