from ANN import ANN, Sigmoid
import numpy as np

# Initialize the ANN with appropriate parameters
ann = ANN([4, 2, 4], [Sigmoid(), Sigmoid()], learning_rate=0.3)

# Set the input and target output
input_data = [1, 0, 0, 0]
target_output = [1, 0, 0, 0]

# Perform feedforward to get function_outputs
function_outputs = ann.feedforward(np.array(input_data))

# Calculate the recommended changes in weights for the output layer
delta_k, delta_w_output = ann.functions[-1].get_output_layer_change(ann, target_output, function_outputs)

# Print the recommended changes in weights for the output layer
print("Recommended changes in weights for the output layer:", delta_w_output)

# Calculate the recommended changes in weights for the hidden layer
delta_hidden = delta_k
for i in range(len(ann.weights) - 2, -1, -1):
    delta_hidden, delta_w_hidden = ann.functions[i].get_hidden_layer_change(ann, function_outputs, i, delta_hidden)
    print(f"Recommended changes in weights for hidden layer {i}: {delta_w_hidden}")