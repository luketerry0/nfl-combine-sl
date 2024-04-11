# classes which represents artificial neural networks and the components which make them up

import numpy as np

class Sigmoid():
    apply = np.vectorize(lambda y: 1/(1 + np.exp(-y)))

    def get_output_layer_change(self, ann, target, function_outputs):
        output = function_outputs[-1]
        negative_ld_net_k = output*(np.ones(len(output)) - output)*(target - output)
        x_i_j = np.insert(function_outputs[-2], 0, 1., axis=0).repeat(len(output))
        delta_w_j_i = np.resize(ann.eta*negative_ld_net_k, x_i_j.shape)

        delta_k = negative_ld_net_k

        return delta_k, delta_w_j_i



    def get_hidden_layer_change(self, ann, function_outputs, layer_idx, previous_delta):
        l_output = np.insert(function_outputs[layer_idx], 0, 1., axis=0)
        oj_term = l_output * (np.ones(l_output.shape) - l_output)
        
        k_term = np.resize(previous_delta, ann.weights[layer_idx].shape)
        multiplied_k_term = k_term*ann.weights[layer_idx]
        sum_k_term = np.resize(multiplied_k_term, (ann.layer_sizes[layer_idx] + 1, ann.layer_sizes[layer_idx + 1])).sum(axis=1)        
        
        delta_j = oj_term*sum_k_term
        xji = np.insert(function_outputs[layer_idx - 1], 0, 1., axis=0)
        resized_xji = xji.repeat(ann.layer_sizes[layer_idx])
        delta_w = ann.eta*np.resize(delta_j[-len(delta_j)+1:], ann.weights[layer_idx - 1].shape)*resized_xji

        return delta_j, delta_w




class ANN():
    def __init__(self, layers : list[int], functions: list, learning_rate: float = 0.3):
        # hyperparameters...
        self.eta = learning_rate
        
        # initialize weights 
        # coordinates for weights are [source_neuron_layer, source_neuron*destination_neuron]
        self.layer_sizes = layers
        self.weights = []

        # add hidden layer weights
        for layer_idx in range(1, len(layers)):
            # initialize weights to very small random values
            self.weights.append(np.random.rand(layers[layer_idx]*(layers[layer_idx - 1] + 1))-0.5*0.1)# (np.random.rand(layers[layer_idx]*layers[layer_idx - 1]) - 0.5)*0.1) #
        
        # keep track of activation functions
        self.functions = functions

    def feedforward(self, inputs : np.array):
        # validate input (protect me from myself....)
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError("Input size does not match the input layer size of the network")
        
        previous_function_outputs = inputs
        function_outputs = [np.array(inputs)]

        for i in range(len(self.layer_sizes) - 1): 
            # calculate the weighted sum of the previous inputs
            # adjust the dimensions of the inputs to multiply elementwise
            previous_function_outputs = np.insert(previous_function_outputs, 0, 1., axis=0).repeat(self.layer_sizes[i + 1])
            weighted_inputs = (previous_function_outputs*self.weights[i]).reshape(self.layer_sizes[i] + 1, self.layer_sizes[i + 1]).sum(axis=0)

            # apply the function
            previous_function_outputs = self.functions[i].apply(weighted_inputs)
            function_outputs.append(previous_function_outputs)            

        
        return(function_outputs)  

    def backpropagate(self, input, target_output):
        # feed the input through the network forwards
        function_outputs = self.feedforward(np.array(input))

        # # change the output weights
        delta, delta_w = self.functions[-1].get_output_layer_change(self, target_output, function_outputs)
        self.weights[-1] += delta_w


        for i in range(len(self.weights) - 1, 0, -1):
            delta, delta_w = self.functions[-2].get_hidden_layer_change(self, function_outputs, i, delta)
            self.weights[i - 1] += delta_w




if __name__ == "__main__":
    # train the ANN based on the example classifier in Mitchell's book
    ann = ANN([4, 2, 4], [Sigmoid(), Sigmoid()], learning_rate=0.1)
    ann.backpropagate([1,0,0,0], [1,0,0,0])

    #ann.backpropagate([1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0])

    EPOCHS = 5000
    for epoch in range(EPOCHS):
        for i in range(4):

            data = [0,0,0,0]
            data[i] = 1

            ann.backpropagate(data, data)


    #after many epochs, check the hidden values of the internal layer
    print("END")
    for i in range(4):

        data = [0,0,0,0]
        data[i] = 1

        results = ann.feedforward(np.array(data))
        print(np.around(results[1]))
        print("         " +str(results[1]))




