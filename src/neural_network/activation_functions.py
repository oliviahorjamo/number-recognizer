import numpy as np


def sigmoid(x_values):
    return 1 / (1 + np.exp(-x_values))


def sigmoid_prime(x_values):
    return np.exp(-x_values) / ((1+np.exp(-x_values))**2)
