import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculate the error of the output layer. Error is a measure (= number)
    of how wrong the output of the layer is.
    
    Parameters:
    y_true: the expected output of the layer
    y_pred: the actual output of the layer
    """
    return np.mean(np.power(y_true-y_pred, 2))