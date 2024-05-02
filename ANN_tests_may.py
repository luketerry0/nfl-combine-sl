from ANN import ANN, Sigmoid, ReLU, MSE_LOSS, CONSTANT_LEARNING_RATE
from preproccessing import Data
import matplotlib.pyplot as plt
import numpy as np

# process the data into training and test data
training_set, test_set = Data.get_data(na_treatment="averaged", standard=True, proportions=[0.9, 0.1])
training_inputs, training_targets = training_set
test_inputs, test_targets = test_set

#create an ANN and train it on the data
EPOCHS = 100
net = ANN([32, 16, 16,  1], [ReLU(), Sigmoid()], MSE_LOSS(), CONSTANT_LEARNING_RATE(0.01), name="test")
training_loss, test_loss = net.train(EPOCHS, training_inputs=training_inputs, training_outputs=training_targets, test_inputs=test_inputs, test_outputs=test_targets)


cm = net.confusion_matrix(test_inputs, test_targets)
print(cm)
acc = (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])
print("accuracy: %s" % acc)

net.ROC_curve(test_inputs, test_targets)
