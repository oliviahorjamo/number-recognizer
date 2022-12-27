import unittest
import index_mnist
import numpy as np
from neural_network.network import Network
from keras.utils import np_utils


class TestIndex(unittest.TestCase):
    def setUp(self):
        return True

    def test_load_and_reshape_data(self):
        (x_train, y_train), (x_test, y_test) = index_mnist.load_data()
        (x_train, y_train), (x_test, y_test) = index_mnist.reshape(x_train,
                                                                    y_train,
                                                                    x_test,
                                                                    y_test)
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
        x_train, y_train, _, _ = index_mnist.create_train_and_test_data()
        net = index_mnist.create_network()
        n_train = 1000
        epochs = 15
        learning_rate = 0.1
        x_train = x_train[0:n_train]
        y_train = y_train[0:n_train]
        epoch_errors = []
        # test that the epoch errors decrease
        for _ in range(epochs):
            epoch_error = net.train_epoch(x_train, y_train, learning_rate)
            epoch_errors.append(epoch_error)
        # sort the errors in a descending order and test that the result doesn't change
        epoch_errors_sorted = sorted(epoch_errors, reverse=True)
        self.assertEqual(epoch_errors_sorted, epoch_errors)
        self.assertLess(epoch_error, 0.05)

    def test_predictions_become_sufficiently_good(self):
        x_train, y_train,x_test,y_test = index_mnist.create_train_and_test_data()
        net = index_mnist.create_network()
        n_train = 1000
        n_test = 100
        epochs = 15
        learning_rate = 0.1
        x_train = x_train[0:n_train]
        y_train = y_train[0:n_train]
        net.train(x_train, y_train, epochs, learning_rate)
        pred = net.predict_multiple(x_test[:n_test])
        corr = index_mnist.correct_values(y_test[:n_test])
        wrong = index_mnist.wrong_indices(pred, corr)
        self.assertLess(len(wrong) / n_test, 0.25)
        

    def test_correct_values(self):
        correct = [1,2,3,4,5,6,7]
        pred = [1,2,0,4,5,6,7]
        wrong_indices_list = index_mnist.wrong_indices(pred, correct)
        self.assertEqual(len(wrong_indices_list), 1)
        self.assertEqual(wrong_indices_list, [2])
        self.assertEqual(pred[wrong_indices_list[0]], 0)        

    
