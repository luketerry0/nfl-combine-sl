# classes which represents artificial neural networks and the components which make them up

import numpy as np

class Sigmoid():
    apply = np.vectorize(lambda y: 1/(1 + np.exp(-y)))

class ANN():
    def __init__(self, layers : list[int], functions: list, learning_rate: float = 0.3):
        # hyperparameters...
        self.eta = learning_rate
        
        
        # initialize weights and biases
        # coordinates for weights are [source_neuron_layer, source_neuron*destination_neuron]
        self.layer_sizes = layers
        self.weights = []

        # add hidden layer weights
        for layer_idx in range(len(layers)):
            if layer_idx != 0:
                # initialize weights to very small random values
                self.weights.append(np.full(layers[layer_idx]*(layers[layer_idx - 1] + 1), 0.0))# (np.random.rand(layers[layer_idx]*layers[layer_idx - 1]) - 0.5)*0.1) #
        

        # keep track of activation functions
        self.functions = functions

    def pass_through_layer(self, idx, input):
        # passes input through a layer

        # add zero to the input
        input = np.insert(input, 0, 1., axis=0)

  
if __name__ == "__main__":
    ann = ANN([4, 2, 4], [Sigmoid(), Sigmoid()], learning_rate=1)
    print(ann.weights)

