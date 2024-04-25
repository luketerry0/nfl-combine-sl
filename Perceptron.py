import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

class PERCEPTRON():
    def __init__(self, num_features, learning_rate, pocket = False):
        # initializes a perceptron with the given number of features
        self.weights = np.zeros(num_features + 1)
        self.pocket = pocket
        self.pocket_misclassifications = math.inf
        if pocket:
            self.pocket_weights = np.zeros(num_features + 1)
        self.eta = learning_rate


    def weighted_sum(self, input : np.array):
        return self.weights.dot(np.insert(input, 0, 1., axis=0))

    def o(self, input: np.array):
        # define the perceptron function
        if self.weighted_sum(input) > 0:
            return 1
        else:
            return -1
        
    def learn(self, input: np.array, target: int, full_training_data : np.array = np.array([])):
        # perceptron learning rule
        o = self.o(input)
        delta_w = self.eta*(target - o)*np.insert(input, 0, 1., axis=0)
        self.weights += delta_w
        
        # if we're running the pocket algorithm, decide whether to update the weights
        if self.pocket:
            runs = np.array([])
            for i in range(10):
                rnd_indices = np.random.choice(len(full_training_data), size=200)
                cm = self.test(full_training_data[rnd_indices])
                runs = np.append(runs, (cm["false_positives"] + cm["false_negatives"]))
            misclassifications = np.mean(runs)
            if misclassifications < self.pocket_misclassifications:
                self.pocket_weights = self.weights
                self.pocket_misclassifications = misclassifications
            else:
                self.weights = self.pocket_weights

    def learn_all(self, data):
        for i in range(0, len(data)):
            self.learn(data[i][:-1], data[i][-1], data)
            if np.isnan(self.weights).any():
                print(data[i])
                raise ValueError("NaN weights from input")
    
    def test(self, data):
        # get the confusion matrix for a given dataset
        cm = {"true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0}
        
        for i in range(len(data)):
            o = self.o(data[i][:-1])
            if o == 1:
                if data[i][-1] == 1:
                    cm["true_positives"] += 1
                else:
                    cm["false_positives"] += 1
            else:
                if data[i][-1] == 1:
                    cm["false_negatives"] += 1
                else:
                    cm["true_negatives"] += 1
        
        return cm



if __name__ == "__main__":
    data = np.genfromtxt('./data/clean_full_data.csv', delimiter=',')[1:] #exclude column names...

    # split data into training and test data
    TEST_PROPORTION = 0.2
    np.random.shuffle(data) 
    training, test = data[:math.floor(len(data)*(1 - TEST_PROPORTION))], data[math.ceil(len(data)*(1 -TEST_PROPORTION)):]

    # run the data through a perceptron, evaluate the confusion matrix

    P = PERCEPTRON(len(training[0][:-1]), 0.1, pocket=False)
    true_positive_rates = []
    precisions = []

    EPOCHS = 100

    for i in range(EPOCHS):
        print(i)
        P.learn_all(training)
        cm = P.test(test)
        if cm["true_positives"] != 0 and cm["false_negatives"] != 0:
            tpr = cm["true_positives"]/(cm["true_positives"] + cm["false_negatives"])
        else: 
            tpr = 0
        if cm["true_positives"] != 0 and cm["false_negatives"] != 0:
            precision = cm["true_positives"]/(cm["true_positives"] + cm["false_positives"])
        else: 
            precision = 0
        true_positive_rates.append(tpr)
        precisions.append(precision)


    plt.plot(range(EPOCHS), true_positive_rates, label = "True Positive Rate")
    plt.plot(range(EPOCHS), precisions, label = "Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Rate")
    plt.legend()
    plt.title("True Positive Rate and Precision (Learning Rate of %s)" % P.eta)
    plt.show()  # display


