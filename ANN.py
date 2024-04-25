import numpy as np
import matplotlib.pyplot as plt

# class encapsulating the sigmoid function
class Sigmoid():
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)
    
# relu activation function
class ReLU():
    def activate(self, x):
        return np.where(x>0,x,0)
        
    def derivative(self, x):
        return np.where(x<=0,0,1)
    
# class encapsulating mean-squared-error loss
class MSE_LOSS():
    def loss(self, predicted_values, target_values):
        return np.mean(0.5*np.square((np.array(predicted_values) - target_values)))
    
    def error(self, _inputs, outputs):
        # derivative of the loss....
        return (_inputs - outputs) 

# cross entropy loss
class CE_LOSS():
    def loss(self, predicted_values, target_values):
        return -1*np.mean(target_values*np.log(predicted_values) + (1 - np.array(target_values))*np.log(1 - np.array(predicted_values)))
    
    def error(self, _inputs, outputs):
        return -1*(np.array(outputs)/np.array(_inputs)) + ((1 - np.array(outputs))/(1 - np.array(_inputs)))

    
# class encapsulating a constant learning rate
class CONSTANT_LEARNING_RATE():
    def __init__(self, value):
        self.learning_rate = value
    def __call__(self, epoch):
        return self.learning_rate
    
# exponentially decaying learning rate
class EXP_DECAY_LEARNING_RATE():
    def __init__(self, init_learning_rate, decay_rate):
        self.decay_rate = decay_rate
        self.init_learning_rate = init_learning_rate
    def __call__(self, epoch):
        return (self.decay_rate**epoch)* self.init_learning_rate

# slower decaying learning rate
class DECAY_LEARNING_RATE():
    def __init__(self, init_learning_rate, decay_rate):
        self.decay_rate = decay_rate
        self.init_learning_rate = init_learning_rate
    def __call__(self, epoch):
        return ((1/(1+self.decay_rate*epoch))*self.init_learning_rate)


# class encapsulating a neural network
# matrix form of backpropigation implemented with mathematical help from https://sudeepraja.github.io/Neural/ rather than the book
class ANN():
    def __init__(self, dimensions: list[int], functions: list, loss_function, learning_rate_function, weights=False, biases=False, name=""):
        self.name = name
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

        # overwrite the weights and biases if some are passed
        if weights and biases:
            self.weights = weights
            self.biases = biases


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
    
    def confusion_matrix(self, x: np.array, y: np.array):
        # makes a confusion matrix based on the passed x and y
        # assumes that we are only doing binary classification...
        cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        print(y[0])
        for i in range(len(x)):
            prediction = round(self.predict(x[i])[0][0])
            if prediction == 1:
                if y[i] == 1:
                    cm["tp"] += 1
                else:
                    cm["fp"] += 1
            else:
                if y[i] == 1:
                    cm["fn"] += 1
                else:
                    cm["tn"] += 1
        return cm
            
    
    def save(self, epoch):
        for i in range(len(self.weights)):
            np.save("previously_trained/weights_%s_%s_%s" % (epoch, i, self.name), self.weights[i])

        for i in range(len(self.biases)):
            np.save("previously_trained/biases_%s_%s_%s" % (epoch, i, self.name), self.biases[i])

            
        

if __name__ == "__main__":
    net = ANN([8, 3, 1], [Sigmoid(), Sigmoid()], CE_LOSS(), CONSTANT_LEARNING_RATE(0.01))

    # build example dataset....
    ds = []
    dy = []
    for i in range(8):
        datapoint = [0,0,0,0,0,0,0,0]
        datapoint[i] = 1
        dy.append(i % 2)
        ds.append(datapoint)


    epochs = 200
    training_loss, test_loss = net.train(epochs, ds, dy, ds, dy)

    plt.plot(range(epochs), training_loss, label="training_loss")
    plt.plot(range(epochs), test_loss, label="test_loss")
    plt.legend()

    plt.show()

    for dp in ds:
        fun = net.forward_pass(dp)
        print(np.round(fun[1]).T)

    print("")
    print(net.confusion_matrix(ds, dy))
    print("")
    
    
    # print(net.weights)

