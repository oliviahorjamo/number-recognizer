import unittest
from neural_network.layer import Layer
import numpy as np

class StubLayer(Layer):
    def __init__(self, input_size, layer_size):
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
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)

    def test_activation_all_negatives(self):
        input_array = np.array([-0.5,-0.5,-0.5])
        output = self.layer.activation(input_array)
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)

    def test_activation_positives_and_negatives(self):
        input_array = np.array([-0.5, 0.5, -0.5])
        output = self.layer.activation(input_array)
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)

    def test_forward_propagation_all_positives(self):
        input_array = np.array([0.5,0.5,0.5])
        output = self.layer.forward_propagation(input_array)
        self.assertNotEqual((output == input_array).all(), True)
        self.assertIsNotNone(self.layer.output)

    def test_forward_propagation_all_negatives(self):
        input_array = np.array([-0.5,-0.5,-0.5])
        output = self.layer.forward_propagation(input_array)
        self.assertNotEqual((output == input_array).all(), True)
        self.assertIsNotNone(self.layer.output)
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)


    def test_forward_propagation_pos_and_neg(self):
        input_array = np.array([-0.5,0.5,-0.5])
        output = self.layer.forward_propagation(input_array)
        self.assertNotEqual((output == input_array).all(), True)
        self.assertIsNotNone(self.layer.output)
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)

    def test_activation_prime_all_positives(self):
        input_array = np.array([1,0.5,1])
        output = self.layer.activation_prime(input_array)
        self.assertEqual((output < input_array).all(), True)

    def test_activation_prime_all_negatives(self):
        input_array = np.array([-0.5,-0.5,-0.5])
        output = self.layer.activation_prime(input_array)
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)

    def test_activation_prime_positives_and_negatives(self):
        input_array = np.array([-0.5, 0.5, -0.5])
        output = self.layer.activation_prime(input_array)
        self.assertEqual((output < np.array([1,1,1])).all(), True)
        self.assertEqual((output > np.array([0,0,0])).all(), True)

    def test_backward_prop_calculate_error_gradient_weights_wrong_output(self):
        self.layer.input = np.array([[1, 1, 1]])
        output_error = np.array([[-0.5, -0.5, -0.5]])
        error_gradient_weights = self.layer.calculate_error_gradient_weights(output_error)
        correct_output = - np.array([[0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5]])
        self.assertEqual((error_gradient_weights == correct_output).all(), True)

    def test_backward_prop_calculate_error_gradient_input(self):
        error_gradient_output = np.array([[0.5, 0.5, 0.5]])
        error_gradient_input = self.layer.calculate_error_gradient_input(error_gradient_output)
        correct_output = np.array([[3/4, 3/4, 3/4]])
        self.assertEqual((error_gradient_input == correct_output).all(), True)

    def test_backward_prop_calculate_error_gradient_weights_right_output(self):
        self.layer.input = np.array([[1, 1, 1]])
        error_gradient_output = np.array([[0, 0, 0]])
        error_gradient_weights = self.layer.calculate_error_gradient_weights(error_gradient_output)
        correct_output = np.zeros((3,3))
        self.assertEqual((error_gradient_weights == correct_output).all(), True)

    def test_backward_prop_calculate_error_gradient_input_right_output(self):
        self.layer.input = np.array([[1, 1, 1]])
        error_gradient_output = np.array([[0, 0, 0]])
        error_gradient_weights = self.layer.calculate_error_gradient_input(error_gradient_output)
        correct_output = np.zeros((3,3))
        self.assertEqual((error_gradient_weights == correct_output).all(), True)

    def test_backward_prop_adjust_weights(self):
        learning_rate = 0.1
        error_gradient_weights = np.array([[0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5]])
        self.layer.backward_propagation_adjust_weights(error_gradient_weights, learning_rate)
        new_weights = np.array([[0.45, 0.45, 0.45],
                                [0.45, 0.45, 0.45],
                                [0.45, 0.45, 0.45]])
        self.assertEqual((self.layer.weights == new_weights).all(), True)

    def test_backward_prop_adjust_biases(self):
        learning_rate = 0.1
        error_gradient_output = np.array([[0.5, 0.5, 0.5]])
        self.layer.backward_propagation_adjust_biases(error_gradient_output, learning_rate)
        new_biases = np.array([[0.45, 0.45, 0.45]])
        self.assertEqual((self.layer.biases == new_biases).all(), True)

    def test_backward_propagation(self):
        learning_rate = 0.1
        input_array = np.array([[1, 1, 1]])
        y_pred = self.layer.forward_propagation(input_array)
        y_true = np.array([[0, 0, 1]])
        error_gradient_output = - (y_true - y_pred)
        weights_first = self.layer.weights
        biases_first = self.layer.biases
        backward_prop_output = self.layer.backward_propagation(error_gradient_output, learning_rate)
        self.assertEqual((self.layer.weights == weights_first).all(), False)
        self.assertEqual((self.layer.biases == biases_first).all(), False)
        self.assertEqual((backward_prop_output == np.zeros((3,1))).all(), False)





