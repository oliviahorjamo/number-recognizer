import numpy as np
# TODO: selvitä miksi tämä rikkoo testit
import activation_functions

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

    def dot_product(self, input_data):
        return np.dot(input_data, self.weights) + self.biases

    def forward_propagation(self, input_data):
        """y = b + x*w where y, b and x are vectors and w is a weight matrix
            for the edges between the previous layer and this layer
        
        Parameters:
        input_data: numpy array
            the output of the previous layer
        """
        x = self.dot_product(input_data)
        x_final = self.activation(x)
        return x_final

    def backward_propagation(self):
        raise NotImplementedError

    def activation(self, input):
        return activation_functions.softmax(input)

layer = Layer(3, 3)
print(layer.weights)
input_array = np.transpose(np.array([1,1,1]))
print(input_array)
print(layer.forward_propagation(input))

