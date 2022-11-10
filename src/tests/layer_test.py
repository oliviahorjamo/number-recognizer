import unittest
from neural_network.layer import Layer
import numpy as np

class StubLayer:
    def __init__(self, input_size, layer_size):
        self.layer = Layer(input_size, layer_size)

class TestLayer(unittest.TestCase):

    def setUp(self):
        self.input_size = 28*28
        self.layer_size = 28*28
        self.layer = Layer(self.input_size, self.layer_size)

    def test_init_layer_size(self):
        self.assertEqual(self.layer.weights.size, self.input_size*self.layer_size)
