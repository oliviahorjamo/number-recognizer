
import numpy as np
import random

from neural_network.network import Network
from neural_network.layer import Layer

from keras.datasets import mnist
from keras.utils import np_utils


def correct_values(y):
    return [np.argmax(array) + 1 for array in y]

def number_of_wrong_answers(y_pred, y_true):
    wrongs = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            wrongs += 1
    return wrongs

def normalize_x(x_train, x_test):
    return x_train / 255, x_test/255

def load_and_reshape():    
    # load MNIST data from keras datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def train_network(net, n_train = 1000, epochs = 15, learning_rate = 0.1):
    net.train(x_train[0:n_train], y_train[0:n_train], epochs=epochs, learning_rate=learning_rate)
    return net

def add_layers(net, layer_sizes = [[28*28, 100], [100, 50], [50, 10]]):
    for layer in layer_sizes:
        net.add_layer(Layer(layer[0], layer[1]))
    return net

def create_network(layer_sizes = [[28*28, 100], [100, 50], [50, 10]]):
    net = Network()

    # always set the size of the input layer to the number of pixels
    # and the size of the output layer to number of categories
    layer_sizes[0][0] = 784
    layer_sizes[len(layer_sizes) - 1][1] = 10

    net = add_layers(net, layer_sizes)
    return net
    
def print_result(pred_classes, true_classes):
    n_wrong = number_of_wrong_answers(pred_classes, true_classes)
    string = (f'predicted values: \n'
            f' {pred_classes} \n'
            f'true values: \n'
            f'{true_classes} \n'
            f'number of wrong answers: \n'
            f'{n_wrong}')
    print(string)

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_and_reshape()

    x_train, x_test = normalize_x(x_train, x_test)

    # one hot encode the output data (from one number to a list of 0s and 1s)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    net = create_network()
    net = train_network(net, epochs=2)

    n_test = 50
    test_indices = random.sample(range(len(x_test)), n_test)

    # test on the random sample of the test data
    pred_classes = net.predict_multiple(x_test[test_indices])
    true_classes = correct_values(y_test[test_indices])

    print_result(pred_classes, true_classes)