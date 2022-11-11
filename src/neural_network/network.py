from neural_network.layer import Layer

class Network:
    """a class to represent the entire network
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input):
        """propagate data forward in the network"""
        for layer in self.layers:
            input = layer.forward_propagation(input)
        # this should be the input after the very last layer
        return input
