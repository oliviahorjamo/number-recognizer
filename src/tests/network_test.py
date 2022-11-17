import unittest
from neural_network.layer import Layer
from neural_network.network import Network
import numpy as np
from tests.layer_test import StubLayer

class StubNetwork():
    def __init__(self):
        self.layers = []

class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = Network()

    def test_add_layer(self):
        # create a layer with input size 3 and output size 3
        # all weights are 1 and so are all biases
        layer = StubLayer(3, 3)
        self.network.add_layer(layer)
        self.assertEqual(len(self.network.layers), 1)

    def test_forward_propagation(self):
        layers = [StubLayer(3, 3), StubLayer(3,3)]
        for layer in layers:
            self.network.add_layer(layer)
        test_data = np.array([1,1,1])
        output = self.network.forward_propagation(test_data)
        self.assertEqual((output == np.array([13, 13, 13])).all(), True)
