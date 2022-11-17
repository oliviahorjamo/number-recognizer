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
