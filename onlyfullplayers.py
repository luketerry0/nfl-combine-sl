# Importing necessary libraries
import pandas as pd
from collections import Counter
# sklearn is only used for confusion matrix and accuracy testing
from sklearn import metrics  
import numpy as np

# Class to represent a node in the decision tree
class Node:
    def __init__(self, is_leaf=False, lbl=None, split_attr=None, branches=None):
        self.is_leaf = is_leaf
        self.lbl = lbl
        self.split_attr = split_attr
        self.branches = branches if branches is not None else {}


# Function to calculate entropy of the given dataset
def calcEntropy(data):
    lbls = data[:, -1]
    counts = np.unique(lbls, return_counts=True)[1]
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob))


# Function to calculate information gain
def calcInfoGain(data, split_idx):
    # Calculate the entropy of the total dataset
    total_ent = calcEntropy(data)
    
    # Calculate the values and the corresponding counts for the split attribute 
    vals, counts = np.unique(data[:, split_idx], return_counts=True)

    # Calculate the weighted entropy
    weighted_ent = np.sum([(counts[i] / np.sum(counts)) * calcEntropy(data[data[:, split_idx] == vals[i]])
                                for i in range(len(vals))])

    # Calculate the information gain
    return total_ent - weighted_ent


# Function to choose the best attribute to split on using information gain
def bestAttr(data, attrs):
    gains = [calcInfoGain(data, attr) for attr in attrs]
    best_attr_idx = np.argmax(gains)
    return attrs[best_attr_idx]

# Function to build the decision tree using ID3 algorithm from the pseudocode
def buildTree(data, attrs, target_idx=-1):
    # If all examples are positive, return a positive leaf node
    if np.all(data[:, target_idx] == 1):
        return Node(is_leaf=True, lbl=1)

    # If all examples are negative, return a negative leaf node
    if np.all(data[:, target_idx] == 0):
        return Node(is_leaf=True, lbl=0)

    # If there are no more attributes to split on, return a leaf node with the most common label
    if len(attrs) == 0:
        most_common = Counter(data[:, target_idx]).most_common(1)[0][0]
        return Node(is_leaf=True, lbl=most_common)

    # Choose the best attribute to split on using information gain
    best_attr = bestAttr(data, attrs)
    attrs = [attr for attr in attrs if attr != best_attr]
    tree_node = Node(split_attr=best_attr)

    # Split the dataset and recursively build the decision tree for each split
    for val in np.unique(data[:, best_attr]):
        subtree_data = data[data[:, best_attr] == val]
        if subtree_data.size == 0:
            most_common = Counter(data[:, target_idx]).most_common(1)[0][0]
            tree_node.branches[val] = Node(is_leaf=True, lbl=most_common)
        else:
            subtree = buildTree(subtree_data, attrs, target_idx)
            tree_node.branches[val] = subtree

    return tree_node


# Function to predict a single instance given the decision tree
def predict(tree, instance):
    while not tree.is_leaf:
        attr_val = instance[tree.split_attr]
        tree = tree.branches.get(attr_val, None)
        if tree is None:
            # Handling the case where the attribute value was not seen during training
            return None
    return tree.lbl


# Function to predict a batch of instances given the decision tree
def predict_batch(tree, instances):
    return [predict(tree, instance) for instance in instances]


# Function to split the dataset into training and testing sets
def split_dataset(data, test_size=0.1, rand_seed=None):
    """
    Split the data into training and testing sets.

    Parameters:
    - data: numpy array, the dataset to split.
    - test_size: float or int, optional (default=0.1), the proportion of the dataset to include in the test split.
    - rand_seed: int or None, optional (default=None), random seed for reproducibility.

    Returns:
    - X_train: numpy array, the training data.
    - X_test: numpy array, the testing data.
    - Y_train: numpy array, the training labels.
    - Y_test: numpy array, the testing labels.
    """
    if rand_seed:
        np.random.seed(rand_seed)

    # Shuffle the data
    np.random.shuffle(data)

    # Determine the number of samples in the test set
    test_samples = int(len(data) * test_size)

    # Split the data
    X_test = data[:test_samples, :-1]
    Y_test = data[:test_samples, -1]
    X_train = data[test_samples:, :-1]
    Y_train = data[test_samples:, -1]

    return X_train, X_test, Y_train, Y_test


# Load the dataset
data = pd.read_csv('allplayers.csv')
data.drop(columns=['Index', 'Year', 'Player', 'Pos'], inplace=True)

# Filter the data to keep only players who competed in all events and have a non-zero value in all columns except the last one
filtered_data = data[(data.iloc[:, :8] > 0).all(axis=1)]
print(filtered_data)


# Assuming the last column is the target (Pick)
target_idx = -1

# Convert the filtered pandas DataFrame to numpy array
filtered_data_array = filtered_data.to_numpy()

# Split the filtered dataset into training and testing sets
X_train, X_test, Y_train, Y_test = split_dataset(filtered_data_array, test_size=0.1, rand_seed=42)

# Build the decision tree based on the training data
attrs = list(range(X_train.shape[1]))  # Exclude the target column
tree = buildTree(np.column_stack((X_train, Y_train)), attrs)

# Example of predicting instances from the testing set
preds = predict_batch(tree, X_test)
print(f"Predictions for testing set: {preds}")

# Convert "None" values to 0 in predictions
mod_preds = [0 if pred is None else pred for pred in preds]

# Calculate accuracy based on modified predictions
acc = metrics.accuracy_score(Y_test, mod_preds)
print(f"Accuracy with modified predictions (Nones=0): {acc}")

# Convert "None" values to 1 in predictions
mod_preds_ones = [1 if pred is None else pred for pred in preds]

# Calculate accuracy based on modified predictions with Nones replaced by 1s
acc_ones = metrics.accuracy_score(Y_test, mod_preds_ones)
print(f"Accuracy with modified predictions (Nones replaced by 1s): {acc_ones}")

# PERCENTAGE OF NONES IN THE PREDICTIONS IS 30.45, SO OVER A QUARTER CANNOT BE PROPERLY PREDICTED.