import unittest
from neural_network.layer import Layer
import numpy as np

class StubLayer:
    def __init__(self, input_size, layer_size):
        self.layer = Layer(input_size, layer_size)

class TestLayer(unittest.TestCase):

    def setUp(self):
        self.input_size = 3
        self.layer_size = 4
        self.layer = Layer(self.input_size, self.layer_size)

    def test_init_layer_weights_size(self):
        self.assertEqual(self.layer.weights.size, self.input_size*self.layer_size)

    def test_init_layer_biases_size(self):
        self.assertEqual(self.layer.biases.size, 1*self.layer_size)

    def test_weights_within_range(self):
        within_range = np.where(np.logical_and(self.layer.weights >= 0, self.layer.weights <= 1))
        self.assertEqual(within_range[0].size, self.layer.weights.size)

