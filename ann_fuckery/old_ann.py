# classes which represents artificial neural networks and the components which make them up

import numpy as np

class Sigmoid():
    apply = np.vectorize(lambda y: 1/(1 + np.exp(-y)))



class ANN():
    def __init__(self, layers : list[int], functions: list, learning_rate: float = 0.1):
        # hyperparameters...
        self.eta = learning_rate
        
        
        # initialize weights and biases
        # coordinates for weights are [source_neuron_layer, source_neuron*destination_neuron]
        # coordinates for biases are [layer, neuron]...
        self.weights = []
        self.biases = []

        for layer_idx in range(len(layers)):
            self.biases.append(np.zeros(layers[layer_idx]))
            if layer_idx != 0:
                # initialize weights to very small random values
                self.weights.append((np.full(layers[layer_idx]*layers[layer_idx - 1], 2)))#np.random.rand(layers[layer_idx]*layers[layer_idx - 1]) - 0.5)*0.1)
        
        # keep track of activation functions
        self.functions = functions

    def feedforward(self, inputs : np.array):

        # validate input (protect me from myself....)
        if len(inputs) != len(self.biases[0]):
            raise ValueError("Input size does not match the input layer size of the network")

        weighted_sums = []
        layer_function_outputs = []

        # feed the input forward through the network
        for layer_idx in range(len(self.biases)):
            # apply the function at each neuron
            fun_outputs = self.functions[layer_idx].apply(inputs)
            layer_function_outputs.append(fun_outputs)

            if layer_idx != len(self.biases) - 1:
                # take a weighted sum to set the input of the next layer
                fun_outputs = np.resize(fun_outputs, len(self.weights[layer_idx]))
                weighted_values = fun_outputs*self.weights[layer_idx]
                inputs = weighted_values.reshape(len(self.biases[layer_idx + 1]), len(self.biases[layer_idx])).sum(axis=1)
                weighted_sums.append(inputs)

        return([fun_outputs, weighted_sums, layer_function_outputs])  

    def SGD_backpropigate(self, input, target_output):
        # compute stochastic gradient descent on the given inputs (including forward propigation)

        # validate output
        if len(target_output) != len(self.biases[-1]):
            raise ValueError("output length differs from final layer length")
        
        output, weighted_sums, layer_outputs = self.feedforward(input)

        # compute delta of the output layer
        delta = np.array(output * (1 - output) * (target_output - output))
        
        # for each layer
        for i in range(len(self.biases) - 2, -1, -1):
            print(delta)
            print(layer_outputs[i])
            # update the weights
            self.weights[i] = self.eta * np.resize(delta,  self.weights[i]) * np.resize(layer_outputs[i], self.weights[i].shape[0])

            # update the new delta
            future_error_sum = delta * np.sum(self.weights[i])
            delta = output * (1 - output) * future_error_sum







if __name__ == "__main__":
    # train the ANN based on the example classifier in Mitchell's book
    ann = ANN([8, 3, 8], [Sigmoid(), Sigmoid(), Sigmoid()])
    ann.SGD_backpropigate([1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0])

'''
    EPOCHS = 5000
    for epoch in range(EPOCHS):
        for i in range(8):

            data = [0,0,0,0,0,0,0,0]
            data[i] = 1

            ann.SGD_backpropigate(data, data)
    
    # after many epochs, check the hidden values of the internal layer
    hidden_values = []
    for i in range(8):

        data = [0,0,0,0,0,0,0,0]
        data[i] = 1

        results = ann.feedforward(data)
        hidden_values.append(results[2][1])
        print(results[2][0])
'''

