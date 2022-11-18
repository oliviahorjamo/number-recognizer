import numpy as np
    
def relu(input_data):
    return (np.maximum(0, input_data))

def relu_item_function(item):
    if item > 0:
        return 1
    return 0

def relu_prime(input_data):
    prime = np.array(list(map(relu_item_function, input_data)))
    return prime
