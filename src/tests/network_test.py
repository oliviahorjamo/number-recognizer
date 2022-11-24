import unittest
from neural_network.layer import Layer
from neural_network.network import Network
from neural_network.loss_functions import mse_gradient
import numpy as np
from tests.layer_test import StubLayer

class StubNetwork():
    def __init__(self):
        self.layers = []

class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = Network()

    def test_add_layer(self):
        layer = StubLayer(3, 3)
        self.network.add_layer(layer)
        self.assertEqual(len(self.network.layers), 1)

    def test_forward_propagation_two_layers(self):
        layers = [StubLayer(3, 3), StubLayer(3,3)]
        for layer in layers:
            self.network.add_layer(layer)
        test_data = np.array([0.5,0.5,0.5])
        output = self.network.forward_propagation(test_data)
        self.assertEqual((output == np.array([2.375, 2.375, 2.375])).all(), True)

    def test_forward_propagation_three_layers(self):
        layers = [StubLayer(3, 3), StubLayer(3,3), StubLayer(3,3)]
        for layer in layers:
            self.network.add_layer(layer)
        test_data = np.array([1,1,1])
        output = self.network.forward_propagation(test_data)
        self.assertEqual((output == np.array([5.75, 5.75, 5.75])).all(), True)

    def test_back_propagation_two_layers_calculate_input_error_gradient(self):
        layers = [StubLayer(3, 3), StubLayer(3,3)]
        for layer in layers:
            self.network.add_layer(layer)
        input_vector = np.array([[0.25, 0.1, 0.5]])
        output = self.network.forward_propagation(input_vector)
        correct_output = np.array([0, 0, 1])
        output_error = mse_gradient(output, correct_output)
        first_layer_input_error_gradient = self.network.backward_propagation(output_error)
        self.assertEqual((
                        first_layer_input_error_gradient == 
                        np.array([[6.99375, 6.99375, 6.99375]]))
                        .all(), 
                        True)


