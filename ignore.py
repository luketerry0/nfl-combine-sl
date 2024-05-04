import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import numpy as np
from collections import Counter

# Function to calculate entropy of the given data set
def entropy(data):
    labels = data[:, -1]
    label_counts = np.unique(labels, return_counts=True)[1]
    probabilities = label_counts / label_counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Function to calculate information gain
def information_gain(data, split_attribute_index):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data)
    
    # Calculate the values and the corresponding counts for the split attribute 
    values, counts = np.unique(data[:, split_attribute_index], return_counts=True)

    # Calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data[data[:, split_attribute_index] == values[i]])
                                for i in range(len(values))])

    # Calculate the information gain
    return total_entropy - weighted_entropy

# Function to choose the best attribute to split on using information gain
def choose_best_attribute(data, attributes):
    information_gains = [information_gain(data, attr) for attr in attributes]
    best_attribute_index = np.argmax(information_gains)
    return attributes[best_attribute_index]

# Class to represent a node in the decision tree
class DecisionTreeNode:
    def __init__(self, is_leaf=False, label=None, split_attribute=None, branches=None):
        self.is_leaf = is_leaf
        self.label = label
        self.split_attribute = split_attribute
        self.branches = branches if branches is not None else {}

# Function to build the decision tree using ID3 algorithm from the pseudocode
def build_decision_tree_id3(data, attributes, target_attribute_index=0):
    # If all examples are positive, return a positive leaf node
    if np.all(data[:, target_attribute_index] == 1):
        return DecisionTreeNode(is_leaf=True, label=1)

    # If all examples are negative, return a negative leaf node
    if np.all(data[:, target_attribute_index] == 0):
        return DecisionTreeNode(is_leaf=True, label=0)

    # If there are no more attributes to split on, return a leaf node with the most common label
    if len(attributes) == 0:
        most_common_label = Counter(data[:, target_attribute_index]).most_common(1)[0][0]
        return DecisionTreeNode(is_leaf=True, label=most_common_label)

    # Choose the best attribute to split on using information gain
    best_attribute = choose_best_attribute(data, attributes)
    attributes = [attr for attr in attributes if attr != best_attribute]
    tree_node = DecisionTreeNode(split_attribute=best_attribute)

    # Split the dataset and recursively build the decision tree for each split
    for value in np.unique(data[:, best_attribute]):
        subtree_data = data[data[:, best_attribute] == value]
        if subtree_data.size == 0:
            most_common_label = Counter(data[:, target_attribute_index]).most_common(1)[0][0]
            tree_node.branches[value] = DecisionTreeNode(is_leaf=True, label=most_common_label)
        else:
            subtree = build_decision_tree_id3(subtree_data, attributes, target_attribute_index)
            tree_node.branches[value] = subtree

    return tree_node

# Function to predict a single instance given the decision tree
def predict(tree, instance):
    while not tree.is_leaf:
        attribute_value = instance[tree.split_attribute]
        tree = tree.branches.get(attribute_value, None)
        if tree is None:
            # Handling the case where the attribute value was not seen during training
            return None
    return tree.label

# Function to predict a batch of instances given the decision tree
def predict_batch(tree, instances):
    return [predict(tree, instance) for instance in instances]

# Example usage with dummy data
# Data format: [attribute1, attribute2, ..., attributeN, target]
dummy_data = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 1]
])

# Build the decision tree based on the dummy data
attributes = list(range(dummy_data.shape[1] - 1))  # Assuming the last column is the target
decision_tree = build_decision_tree_id3(dummy_data, attributes)


# Predict the class for a new instance using the built decision tree
new_instance = np.array([1, 0])
prediction = predict(decision_tree, new_instance)
print(f"Prediction for {new_instance}: {prediction}")
