import unittest
import index_mnist
import numpy as np
from neural_network.network import Network
from keras.utils import np_utils


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

    def test_epoch_errors_decrease_and_and_become_sufficiently_small(self):
        (x_train, y_train), (x_test, y_test) = index_mnist.load_and_reshape()
        x_train, x_test = index_mnist.normalize_x(x_train, x_test)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        net = index_mnist.create_network()
        n_train = 1000
        epochs = 15
        learning_rate = 0.1
        x_train = x_train[0:n_train]
        y_test = y_train[0:n_train]
        epoch_errors = []
        for i in range(epochs):
            epoch_error = net.train_epoch(x_train, y_train, learning_rate)
            epoch_errors.append(epoch_error)
        # sort the errors in a descending order and test that the result doesn't change
        epoch_errors_sorted = sorted(epoch_errors, reverse=True)
        self.assertEqual(epoch_errors_sorted, epoch_errors)
        self.assertLess(epoch_error, 0.05)
