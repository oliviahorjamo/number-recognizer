import unittest
import index_mnist
import numpy as np
from neural_network.network import Network

class TestIndex(unittest.TestCase):
    def setUp(self):
        return True

    def test_load_data(self):
        (x_train, y_train), (x_test, y_test) = index_mnist.load_and_reshape()
        self.assertEqual(len(x_train), 60000)
        self.assertEqual(len(x_test), 10000)
        self.assertEqual(len(y_train), 60000)
        self.assertEqual(len(y_test), 10000)
        self.assertEqual(x_test.dtype, 'float32')
        self.assertEqual(x_train.dtype, 'float32')

    def test_normalize_data(self):
        x_train = np.array([0, 100, 250])
        x_test = np.array([10, 25, 50])
        x_train, x_test = index_mnist.normalize_x(x_train, x_test)
        self.assertEqual((x_train == [0, 100/255, 250/255]).all(), True)
        self.assertEqual((x_test == [10/255, 25/255, 50/255]).all(), True)

    def test_create_network(self):
        net = index_mnist.create_network()
        self.assertEqual(isinstance(net, Network), True)
        self.assertEqual(net.layers[0].weights.shape[0], 784)
        self.assertEqual(net.layers[len(net.layers) - 1].weights.shape[1], 10)
