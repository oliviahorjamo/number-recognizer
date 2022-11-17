
import numpy as np

def softmax_item(item):
    return max(0, item)

def softmax_old(input_data):
    func = np.vectorize(softmax_item)
    return func(input_data)
    
def relu(input_data):
    return (np.maximum(0, input_data))