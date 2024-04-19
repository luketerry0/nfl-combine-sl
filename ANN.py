import numpy as np
import matplotlib.pyplot as plt

# class encapsulating the sigmoid function
class Sigmoid():
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)


# class encapsulating a neural network
# matrix form of backpropigation implemented with mathematical help from https://sudeepraja.github.io/Neural/ rather than the book
class ANN():
    def __init__(self, dimensions: list[int], functions: list):
        # learning rate
        self.eta = 0.01

        # initalize weight matrices and bias vectors
        self.funs = functions
        self.dims = dimensions
        self.weights = []
        self.biases = []
        for i in range(len(dimensions) - 1):
            self.weights.append(np.random.randn(dimensions[i + 1], dimensions[i]))
            self.biases.append(np.array([np.zeros(dimensions[i + 1])]).T)


    def forward_pass(self, input: list[float]):
        # feed a value forwards through the network
        input = np.array([input]).T
        fun_outputs = [input]
        for i in range(len(self.dims) - 1):
            input = self.funs[i].activate(np.dot(self.weights[i], input) + self.biases[i])
            fun_outputs.append(input)        

        return(fun_outputs)
    
    def backwards_pass(self, target, fun_outputs):
            target = np.array([target]).T
            deltas = [(fun_outputs[-1] - target) * self.funs[-1].derivative(fun_outputs[-1])]

            for i in range(len(self.dims) - 3, -1, -1):
                # Calculate derivative of the activation function for the current layer
                activation_derivative = self.funs[i].derivative(fun_outputs[i+1])

                # Calculate delta for the current layer
                delta = np.dot(self.weights[i+1].T, deltas[0]) * activation_derivative
                deltas.insert(0, delta)
            
            return deltas


    def backpropagate(self, input, target):
        fun_outs = self.forward_pass(input)
        delta = self.backwards_pass(target, fun_outs)

        for i in range(len(self.weights)):
            # update weights....
            loss_der = np.matmul(delta[i], fun_outs[i].T)
            self.weights[i] -= loss_der*self.eta

            # update biases....
            self.biases[i] -= delta[i]*self.eta
        
    def getLoss(self, inputs):
        losses = []
        for _input in inputs:
            result = self.forward_pass(_input)
            losses.append(np.sum(0.5*np.square(np.array(_input) - result[-1])))
        
        return np.average(losses)

if __name__ == "__main__":
    net = ANN([4, 2, 4], [Sigmoid(), Sigmoid()])
    # net.backpropagate([1,0,0,0], [1,0,0,0])
    # net.backpropagate([1,0,0,0], [1,0,0,0])

    epochs = 50000
    loss = []
    for i in range(epochs):
        inputs = []
        for j in range(4):
            ip = [0,0,0,0]
            ip[j] = 1
            inputs.append(ip)
            net.backpropagate(ip, ip)
        ls = net.getLoss(inputs)
        loss.append(ls)
    # print(net.forward_pass([1, 0,0,0]))
    # print("-=")
    # print(net.weights)

    #plt.plot(range(epochs), loss)

    #plt.show()

    for i in range(4):
        ip = [0,0,0,0]
        ip[i] = 1
        fun = net.forward_pass(ip)
        print(fun[1])
    print("")
    
    
    # print(net.weights)

