import unittest
from neural_network.layer import Layer
import numpy as np

class StubLayer(Layer):
    def __init__(self, input_size, layer_size):
        #self.layer = Layer(input_size, layer_size)
        super().__init__(input_size, layer_size)
        self.weights = np.ones((input_size, layer_size)) - 0.5
        self.biases = np.ones((1, layer_size)) -0.5

class TestLayer(unittest.TestCase):

    def setUp(self):
        self.input_size = 3
        self.layer_size = 3
        self.layer = StubLayer(self.input_size,self.layer_size)

    def test_init_layer_weights_size(self):
        self.assertEqual(self.layer.weights.size, self.input_size*self.layer_size)

    def test_init_layer_biases_size(self):
        self.assertEqual(self.layer.biases.size, 1*self.layer_size)

    def test_weights_within_range(self):
        within_range = np.where(np.logical_and(self.layer.weights >= 0, self.layer.weights <= 1))
        self.assertEqual(within_range[0].size, self.layer.weights.size)

    def test_dot(self):
        input_array = np.array([1,1,1])
        output = self.layer.dot_product(input_array)
        self.assertEqual((output == np.array([3/2,3/2,3/2])).all(), True)

    def test_add_biases(self):
        input_array = np.array([1,1,1])
        output = self.layer.add_biases(input_array)
        self.assertEqual((output == np.array([1.5,1.5,1.5])).all(), True)

    def test_activation_all_positives(self):
        input_array = np.array([1,0.5,1])
        output = self.layer.activation(input_array)
        self.assertEqual((output == np.array([1,0.5,1])).all(), True)

    def test_activation_all_negatives(self):
        input_array = np.array([-0.5,-0.5,-0.5])
        output = self.layer.activation(input_array)
        self.assertEqual((output == np.array([0,0,0])).all(), True)

    def test_activation_positives_and_negatives(self):
        input_array = np.array([-0.5, 0.5, -0.5])
        output = self.layer.activation(input_array)
        self.assertEqual((output == np.array([0, 0.5, 0])).all(), True)

    def test_forward_propagation_all_positives(self):
        input_array = np.array([0.5,0.5,0.5])
        output = self.layer.forward_propagation(input_array)
        self.assertEqual((output == np.array([5/4,5/4,5/4])).all(), True)

    def test_forward_propagation_all_negatives(self):
        input_array = np.array([-0.5,-0.5,-0.5])
        output = self.layer.forward_propagation(input_array)
        self.assertEqual((output == np.array([0,0,0])).all(), True)

    def test_forward_propagation_pos_and_neg(self):
        input_array = np.array([-0.5,0.5,-0.5])
        output = self.layer.forward_propagation(input_array)
        self.assertEqual((output == np.array([0.25,0.25,0.25])).all(), True)

    def test_activation_prime_all_positives(self):
        input_array = np.array([1,0.5,1])
        output = self.layer.activation_prime(input_array)
        self.assertEqual((output == np.array([1,1,1])).all(), True)

    def test_activation_prime_all_negatives(self):
        input_array = np.array([-0.5,-0.5,-0.5])
        output = self.layer.activation_prime(input_array)
        self.assertEqual((output == np.array([0,0,0])).all(), True)

    def test_activation_prime_positives_and_negatives(self):
        input_array = np.array([-0.5, 0.5, -0.5])
        output = self.layer.activation_prime(input_array)
        self.assertEqual((output == np.array([0, 1, 0])).all(), True)



