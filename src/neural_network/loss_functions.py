import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate the error of the output layer. Error is a measure (= number)
    of how wrong the output of the layer.
    
    Parameters:
    y_true: the expected output of the layer
    y_pred: the actual output of the layer
    """
    error_vector = (y_true - y_pred) * (y_true - y_pred)
    error = 1 / len(y_true) * sum(error_vector)
    return error

def mse_gradient(y_true, y_pred):
    """Calculate the gradient of the error with respect to the output. Gradient
    tells the derivative of the error with respect to each neuron i.e., tells how
    wrong the output of each neuron is.
    """
    error_vector = y_true - y_pred
    print('error vector')
    print(error_vector)
    gradient = 2 / len(y_true) * error_vector
    return gradient
