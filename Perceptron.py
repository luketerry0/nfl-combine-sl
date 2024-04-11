import numpy as np
class PERCEPTRON():
    def __init__(self, num_features, threshold):
        # initializes a perceptron with the given number of features
        self.weights = np.zeros(num_features + 1)