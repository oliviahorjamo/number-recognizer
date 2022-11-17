import numpy as np
from neural_network import activation_functions


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
        output_size: the number of neurons in this layer

        The columns represent the weights of the edges between one neuron in this
        layer and all neurons in the previous layer. Hence, each column represents one
        neuron in this layer. Since there is only one bias term per neuron
        the size of self.biases is (1, layer_size).
        """

        self.weights = np.random.rand(input_size, layer_size)
        self.biases = np.random.rand(1, layer_size)

    def dot_product(self, input_data):
        return np.dot(input_data, self.weights)

    def add_biases(self, input_data):
        return input_data + self.biases

    def forward_propagation(self, input_data):
        """y = b + x*w where y, b and x are vectors and w is a weight matrix
            for the edges between the previous layer and this layer

        Parameters:
        input_data: numpy array
            the output of the previous layer
        """
        x_dot_weigths = self.dot_product(input_data)
        x_plus_biases = self.add_biases(x_dot_weigths)
        x_final = self.activation(x_plus_biases)
        return x_final

    def backward_propagation(self):
        raise NotImplementedError

    def activation(self, input_array):
        return activation_functions.softmax(input_array)
