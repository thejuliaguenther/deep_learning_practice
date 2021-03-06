import numpy as np

# Practice implementing a forward pass through a 4x3x2 network,
# with sigmoid activation functions for both layers.
#
# Based on an exercise from Udacity.com
###################################################################
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1 + np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)

# Make some fake data
X = np.random.randn(4)

weights_in_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_out = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

hidden_layer_in = np.dot(X, weights_in_hidden)
hidden_layer_out =sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_out)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)