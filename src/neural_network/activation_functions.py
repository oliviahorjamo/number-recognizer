import numpy as np


def relu(input_data):
    return (np.maximum(0, input_data))

def relu_item_function(item):
    print('item in item function')
    print(item)
    if item > 0:
        return 1
    return 0


def relu_prime(input_data):
    """Return the derivate of the activation function for this input data.
    Because the activation function returns the value itself if it is larger than 0,
    the derivative is 1 for values larger than 0. Otherwise the derivative is 0,
    because the activation function maps these values to 0."""
    prime = np.where(input_data > 0, 1, 0)
    return prime


def sigmoid(x_values):
    return 1 / (1 + np.exp(-x_values))


def sigmoid_prime(x_values):
    return np.exp(-x_values) / ((1+np.exp(-x_values))**2)
