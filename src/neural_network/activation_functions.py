
import numpy as np

def softmax_item(item):
    return max(0, item)

def softmax(input_data):
    func = np.vectorize(softmax_item)
    return func(input_data)
    