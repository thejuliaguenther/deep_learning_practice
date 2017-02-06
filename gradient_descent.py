import numpy as np

<<<<<<< HEAD
# Practice implementing one gradient descent step for the weights
# of a simple network with two inputs and one output unit with a sigmoid
# activation function
#
# Based on an exercise from Udacity.com
###################################################################

=======
# Practice implementing Gradient Descent based on Udacity example 
>>>>>>> 5d5060db4eac18cc771934beee84dca08524240a
def sigmoid(x):
    """
    Calculates sigmoid
    """
    return 1/(1+np.exp(-x))
    
def sigmoid_prime(x):
    """
    Calculates the derivative of the sigmoid function
    """
    f_h = sigmoid(x)
    result = f_h * (1 - f_h)
    
    return result

learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

nn_output = sigmoid(np.dot(x,w))

error = y - nn_output

error_gradient = error * sigmoid_prime(np.dot(x,w))

del_w = learnrate * error_gradient * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)