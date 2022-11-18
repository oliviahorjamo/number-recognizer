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

        self.weights = np.random.rand(input_size, layer_size) - 0.5
        self.biases = np.random.rand(1, layer_size) - 0.5
        self.input = None

    def dot_product(self, input_data):
        return np.dot(input_data, self.weights)

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

    def backward_propagation(self, error_gradient_output):
        """Based on the error with respect to the output of this layer, calculate the error
        with respect to the weights and biases of this layer and then output the error with
        respect to the input of this layer. This will yield the output_error for the previous
        layer.
        
        Parameters:
        error_gradient_output: the gradient of the error with respect to this layer's output

        Outputs:
        error_gradient_inputs: the derivative of the error with respect to the input of this layer
        
        """

        # should be of size (input_size, layer_size since there is one gradient for every weight)
        # the partial derivative of E with respect to w can be calculated with the
        # chain rule dE/dw = dE/dy * dy/dw
        # because y = xw + b, dy/dw = x
        error_gradient_weights = np.dot(self.input.T, error_gradient_output)
        self.backward_propagation_adjust_weights(error_gradient_weights)

        # shold be of size (1, layer_size) since there is one gradient for each bias
        # y = xw + b --> dE/db = dE/dy * dy/db = dE/dy (=output_error)
        error_gradient_biases = error_gradient_output
        self.backward_propagation_adjust_biases(error_gradient_biases)

        # should be of size (1, input_error)
        # y = xw + b --> dE / dx = dE / dy * dy/dx = dE/dy* w
        # serves as the error with respect to the output for the previous layer
        error_gradient_inputs = np.dot(error_gradient_output, self.weights)

        return error_gradient_inputs

    def backward_propagation_adjust_weights(self, error_gradient_weights):
        """Adjust the weights of the edges between the previous layer and this layer
        by the gradient of the error with respect to weights.

        Parameters:
        error_gradient_weights: the gradient of the error with respect to weights. The
        negative of the gradient tells the direction of deepest decrease, i.e. which weights
        to adjust and by how much to decrease the error the fastest."""
        self.weights = self.weights - error_gradient_weights

    def backward_propagation_adjust_biases(self, error_gradient_biases):
        """Adjust the biases of the neurons in this layer by the gradient of the error
        with respect to biases.

        Parameters:
        error_gradient_biases: the gradient of the error with respect to biases. The
        negative of the gradient tells the direction of deepest decrease, i.e. which biases
        to adjust and by how much to decrease the error the fastest."""
        self.biases = self.biases - error_gradient_biases
