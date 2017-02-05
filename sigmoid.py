import numpy as np

def sigmoid(x):
    """
    This function calculates the output of a simple neural network 
    of 2 nodes with a sigmoid activation function
    """
    result = 1/(1 + np.exp(-x))
    return result

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

output = sigmoid(np.dot(inputs, weights) + bias)

print('Output:')
print(output)
