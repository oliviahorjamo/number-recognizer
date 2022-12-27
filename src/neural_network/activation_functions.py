import numpy as np


def sigmoid(x_values):
    """Apply an activation function to the input data so that
    the network can perform other than non-linear tasks.
    
    Parameters:
    x-values: An input vector that contains the output of this layer
    before applying the activation function.
    
    Returns:
    The same vector passed through a sigmoid function that squeezes
    all values to a [0,1] range.
    """
    return 1 / (1 + np.exp(-x_values))


def sigmoid_prime(x_values):
    """Apply the derivative of the sigmoid function to the input data.
    Returns the input data back to what it was before applying the activation
    function.
    
    Parameters:
    x_values: A vector that contains the output of this layer after
    applying the activation function.
    
    Returns:
    The derivative of the activation function with the given values.
    """
    return np.exp(-x_values) / ((1+np.exp(-x_values))**2)
