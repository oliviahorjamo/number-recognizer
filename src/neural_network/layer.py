import numpy as np

class Layer:
    """A class that represents a layer
    
    Attributes:
    weights: numpy array
        the weights of the edges between the neurons in this layer and the neurons in previous layer    
    bias: numpy array
        the biases of the neurons of the layer
    """

    def __init__(self, input_size, layer_size):
        """creates a new Layer -class instance with random weigths and biases
        
        Parameters:
        input_size: the size of the input given to this layer
        output_size: the number of neurons in this layer"""

        self.weights = np.random.rand(input_size, layer_size)
        self.biases = np.random.rand(layer_size, 1)