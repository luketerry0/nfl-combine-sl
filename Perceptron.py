import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from preproccessing import Data
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

class PERCEPTRON():
    def __init__(self, num_features, learning_rate, pocket = False):
        # initializes a perceptron with the given number of features
        self.weights = np.full(num_features + 1, 0)
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

        # test the number of misclassifications on the whole dataset, and decide whether to update the pocket
        if self.pocket:
            pred = np.sign(np.array(full_training_data[:,:-1] @ (self.weights)[:-1] + (self.weights)[-1]))
            pred[pred == 0] = 1
            n_misclassifications = np.sum(pred != full_training_data[:,-1:].T)
            if n_misclassifications < self.pocket_misclassifications:
                print("nmis %s" % n_misclassifications)
                self.pocket_misclassifications = n_misclassifications
                self.weights = self.weights + delta_w

        else:
            self.weights = self.weights + delta_w
        # cm = self.test(full_training_data)
        # acc = (cm["tp"] + cm["tn"])/(cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"])
        # print("t_acc %s" % acc)
        
                
    def learn_all(self, data):
        if self.pocket & (self.pocket_misclassifications == math.inf):
            pred = np.sign(np.array(data[:,:-1] @ (self.weights)[:-1] + (self.weights)[-1]))
            pred[pred == 0] = 1
            self.pocket_misclassifications = np.sum(pred != data[:,-1:].T)
        print(self.pocket_misclassifications)

        for i in range(0, len(data)):
            updated = self.learn(data[i][:-1], data[i][-1], data)
            if updated:
                print(self.weights)
            
            if np.isnan(self.weights).any():
                print(data[i])
                raise ValueError("NaN weights from input")
            

    
    def test(self, data):
        # get the confusion matrix for a given dataset
        cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        
        for i in range(len(data)):
            o = self.o(data[i][:-1])
            if o == 1:
                if data[i][-1] == 1:
                    cm["tp"] += 1
                else:
                    cm["fp"] += 1
            else:
                if data[i][-1] == 1:
                    cm["fn"] += 1
                else:
                    cm["tn"] += 1
        
        return cm



if __name__ == "__main__":
    training, test = Data.get_data(na_treatment="averaged", proportions=[0.9,0.1], standard=True, negative_falses=True, seed=28347209)
 
    print("negative proportion %s" % str(np.sum(np.array(training[1]) != 1)/len(training[1])))

    training = np.hstack((training[0], np.array([training[1]]).T))
    test = np.hstack((test[0], np.array([test[1]]).T))
    

    
    # run the data through a perceptron, evaluate the confusion matrix
    P = PERCEPTRON(len(training[0][:-1]), 1, pocket=True)
    accuracy = []

    EPOCHS = 100

    for i in range(EPOCHS):
        print("")
        print("epoch %s" % i)
        P.learn_all(training)
        cm = P.test(test)
        
        acc = (cm["tp"] + cm["tn"])/(cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"])
        print(acc)
        accuracy.append(acc)
        # if cm["true_positives"] != 0 and cm["false_negatives"] != 0:
        #     tpr = cm["true_positives"]/(cm["true_positives"] + cm["false_negatives"])
        # else: 
        #     tpr = 0
        # if cm["true_positives"] != 0 and cm["false_negatives"] != 0:
        #     precision = cm["true_positives"]/(cm["true_positives"] + cm["false_positives"])
        # else: 
        #     precision = 0
        # true_positive_rates.append(tpr)
        # precisions.append(precision)


    plt.plot(range(EPOCHS), accuracy, label = "Accuracy")
    plt.axhline(y = np.mean(accuracy), color = 'orange', linestyle = '-', label = "Mean Accuracy") 
    plt.xlabel("Epochs")
    plt.ylabel("Percent Accuracy")
    plt.legend()
    plt.title("Perceptron Accuracy (Learning Rate of %s)" % P.eta)
    plt.show()  # display

    print("MEAN: %s" % np.mean(accuracy))

    # save the weights just incase...
    np.save("./perceptron_weights.npx", P.weights)


