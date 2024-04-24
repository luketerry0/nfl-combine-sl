from ANN import ANN, Sigmoid, MSE_LOSS, CONSTANT_LEARNING_RATE
from preproccessing import Data
import matplotlib.pyplot as plt
import numpy as np

# process the data into training and test data
training_set, test_set = Data.get_data(na_treatment="averaged", proportions=[0.9, 0.1])
training_inputs, training_targets = training_set
test_inputs, test_targets = test_set

# create an ANN and train it on the data
EPOCHS = 10000
net = ANN([32, 10, 1], [Sigmoid(), Sigmoid(), Sigmoid()], MSE_LOSS(), CONSTANT_LEARNING_RATE(0.01))
training_loss, test_loss = net.train(EPOCHS, training_inputs=training_inputs, training_outputs=training_targets, test_inputs=test_inputs, test_outputs=test_targets)



plt.plot(range(EPOCHS), training_loss, label="training loss")
plt.plot(range(EPOCHS), test_loss, label="test loss")
plt.legend()

plt.show()



