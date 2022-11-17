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

    def dot_product_transpose(self, input_data):
        return np.dot(input_data, self.weights.T)

    def add_biases(self, input_data):
        return input_data + self.biases
    
    def activation(self, input_array):
        return activation_functions.relu(input_array)

    def activation_prime(self, input_array):
        return activation_functions.relu_prime(input_array)

    def forward_propagation(self, input_array):
        """Propagate data forwards: calculate the output of this layer by calculating the
        dot product of the input data and the weights between this layer and the previous
        layer and add biases. Save input_data as self.input since this information will be
        needed later in back propagation.

        Parameters:
        input_data: numpy array
            the output of the previous layer
        """
        self.input = input_array
        x_dot_weigths = self.dot_product(input_array)
        x_plus_biases = self.add_biases(x_dot_weigths)
        x_final = self.activation(x_plus_biases)
        return x_final

    def backward_propagation(self, output_error):
        """Based on the error with respect to the output of this layer, calculate the error
        with respect to the weights and biases of this layer and then output the error with
        respect to the input of this layer. This will yield the output_error for the previous
        layer.
        
        Parameters:
        output_error: the derivate of the error with respect to this layer's output

        Outputs:
        the derviate of the error with respect to the input of this layer
        
        """
        input_error = self.dot_product_transpose(output_error)
