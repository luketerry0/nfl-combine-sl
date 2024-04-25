import numpy as np
import matplotlib.pyplot as plt

# class encapsulating the sigmoid function
class Sigmoid():
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)
    
# class encapsulating mean-squared-error loss
class MSE_LOSS():
    def loss(self, predicted_values, target_values):
        return np.mean(0.5*np.square((np.array(predicted_values) - target_values)))
    
    def error(self, _inputs, outputs):
        # derivative of the loss....
        return (_inputs - outputs) 
    
# class encapsulating a constant learning rate
class CONSTANT_LEARNING_RATE():
    def __init__(self, value):
        self.learning_rate = value
    def __call__(self, epoch):
        return self.learning_rate



# class encapsulating a neural network
# matrix form of backpropigation implemented with mathematical help from https://sudeepraja.github.io/Neural/ rather than the book
class ANN():
    def __init__(self, dimensions: list[int], functions: list, loss_function, learning_rate_function):
        # learning rate
        self.eta = learning_rate_function(1)
        self.learning_rate_function = learning_rate_function
        self.loss_function = loss_function

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
            deltas = [self.loss_function.error(fun_outputs[-1],  target)* self.funs[-1].derivative(fun_outputs[-1])]

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
        
        # return the training loss from this example
        return self.loss_function.loss(input, fun_outs[-1])
        
    def predict(self, input):
        return self.forward_pass(input)[-1]
    
    def train(self, epochs, training_inputs, training_outputs, test_inputs, test_outputs):
        training_loss = []
        test_loss = []
        for epoch in range(epochs):

            # update the learning rate (in case it is altered)
            self.eta = self.learning_rate_function(epoch)

            # train and return the average training and test loss
            epoch_training_loss = []
            epoch_test_loss = []
            for i in range(len(training_inputs)):
                self.backpropagate(training_inputs[i], training_outputs[i])
                ep_training_loss = self.loss_function.loss(self.predict(training_inputs[i]), training_outputs[i])
                epoch_training_loss.append(ep_training_loss)

            for i in range(len(test_inputs)):
                ep_test_loss = self.loss_function.loss(self.predict(test_inputs[i]), test_outputs[i])
                epoch_test_loss.append(ep_test_loss)

            training_loss.append(np.mean(epoch_training_loss))
            test_loss.append(np.mean(epoch_test_loss))


            if epoch % 100 == 0:
                print("completed %s epochs" % str(epoch))
                print("TEST LOSS    : %s" % np.mean(epoch_test_loss))
                print("TRAINING LOSS: %s" % np.mean(epoch_training_loss))
                self.save(epochs)


        return [training_loss, test_loss]
    
    def save(self, epoch):
        for i in range(len(self.weights)):
            np.save("previously_trained/weights_%s_%s" % (epoch, i), self.weights[i])

        for i in range(len(self.biases)):
            np.save("previously_trained/biases_%s_%s" % (epoch, i), self.biases[i])

            
        

if __name__ == "__main__":
    net = ANN([8, 3, 8], [Sigmoid(), Sigmoid(), Sigmoid()], MSE_LOSS(), CONSTANT_LEARNING_RATE(0.01))

    # build example dataset....
    ds = []
    for i in range(8):
        datapoint = [0,0,0,0,0,0,0,0]
        datapoint[i] = 1
        ds.append(datapoint)


    epochs = 2000
    training_loss, test_loss = net.train(epochs, ds, ds, ds, ds)

    plt.plot(range(epochs), training_loss, label="training_loss")
    plt.plot(range(epochs), test_loss, label="test_loss")
    plt.legend()

    plt.show()

    for dp in ds:
        fun = net.forward_pass(dp)
        print(np.round(fun[1]).T)
    print("")
    
    
    # print(net.weights)

