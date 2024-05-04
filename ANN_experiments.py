from ANN import ANN, Sigmoid, ReLU, MSE_LOSS, CONSTANT_LEARNING_RATE, CE_LOSS
from preproccessing import Data
import matplotlib.pyplot as plt
import numpy as np

# process the data into training and test data
training_set, test_set = Data.get_data(na_treatment="averaged", proportions=[0.9, 0.1], standard=True)
training_inputs, training_targets = training_set
test_inputs, test_targets = test_set

#create an ANN and train it on the data
EPOCHS = 300000
name = "mse_ted"
net = ANN([32, 16, 16, 1], [ ReLU(), ReLU(), Sigmoid()], MSE_LOSS(), CONSTANT_LEARNING_RATE(0.00001), name=name)
training_loss, test_loss = net.train(EPOCHS, training_inputs=training_inputs, training_outputs=training_targets, test_inputs=test_inputs, test_outputs=test_targets)

np.save("./previously_trained/training_loss_%s" % name, np.array(training_loss))
np.save("./previously_trained/test_loss_%s" % name, np.array(test_loss))


plt.plot(range(EPOCHS), training_loss, label="training loss")
plt.plot(range(EPOCHS), test_loss, label="test loss")
plt.legend()

plt.show()
# weights = [np.load("./previously_trained/weights_300000_0_ted.npy"), np.load("./previously_trained/weights_300000_1_ted.npy"), np.load("./previously_trained/weights_300000_2_ted.npy")]
# biases = [np.load("./previously_trained/biases_300000_0_ted.npy"), np.load("./previously_trained/biases_300000_1_ted.npy"), np.load("./previously_trained/weights_300000_2_ted.npy")]


# net = ANN([32, 16, 16, 1], [ReLU(), ReLU(), Sigmoid()], MSE_LOSS(), CONSTANT_LEARNING_RATE(0.01), weights=weights, biases=biases)
cm = net.confusion_matrix(test_inputs, test_targets)
print(cm)
print("")
print("Accuracy: %s" % str((cm["tp"] + cm["tn"])/(cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"])))


