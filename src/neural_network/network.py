#from neural_network.layer import Layer

class Network:
    """a class to represent the entire network
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input_array):
        """propagate data forward in the network"""
        for layer in self.layers:
            input_array = layer.forward_propagation(input_array)
        return input_array

    def backward_propagation(self, output_error):
        """propagate data backwards in the network
        
        Note that the for loop must go through the layers in reversed
        order since the data is flowing from the last layer to the first layer."""
        for layer in reversed(self.layers):
            output_error = layer.backward_propagation(output_error)
        return output_error
        