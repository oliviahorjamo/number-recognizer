import unittest
import numpy as np
import neural_network.loss_functions as loss_functions

class TestError(unittest.TestCase):

    def setUp(self):
        pass

    def test_mse_correct_output(self):
        y_true = np.array([1,1,1])
        y_pred = np.array([1,1,1])
        self.assertEqual(loss_functions.mean_squared_error(y_true, y_pred), 0.0)

    def test_mse_wrong_output(self):
        y_true = np.array([1,0,0])
        y_pred = np.array([0.5,0.5,0.5])
        self.assertEqual(loss_functions.mean_squared_error(y_true, y_pred), 0.25)

    def test_mse_wrong_output_random(self):
        y_true = np.random.rand(3,)
        y_pred = np.random.rand(3,)
        self.assertNotEqual(loss_functions.mean_squared_error(y_true, y_pred), 0)
