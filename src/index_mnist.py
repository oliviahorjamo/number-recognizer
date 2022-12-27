import random
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from neural_network.network import Network
from neural_network.layer import Layer
import matplotlib.pyplot as plt


def correct_values(y_true):
    return [np.argmax(array) for array in y_true]


def wrong_indices(y_pred, y_true):
    """Return the indices of the wrongly categorized numbers."""
    wrong_indices = []
    for i, value in enumerate(y_pred):
        if value != y_true[i]:
            wrong_indices.append(i)
    return wrong_indices


def normalize_x(x_train, x_test):
    return x_train / 255, x_test/255


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def reshape(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)


def train_network(net, x_train, y_train, n_train=1000, epochs=15, learning_rate=0.1):
    last_error = net.train(
        x_train[0:n_train], y_train[0:n_train], epochs=epochs, learning_rate=learning_rate)
    return net, last_error


def add_layers(net, layer_sizes=[[28*28, 100], [100, 50], [50, 10]]):
    for layer in layer_sizes:
        net.add_layer(Layer(layer[0], layer[1]))
    return net


def create_network(layer_sizes=[[28*28, 100], [100, 50], [50, 10]]):
    """Create a network with the given layer sizes.
    
    Parameters:
    layer_sizes: a list of lists of length 2 where each list contains
    the sizes of the layer at this index
    
    Returns:
    net: an object of the Network class with the given layers
    """
    net = Network()

    # always set the size of the input layer to the number of pixels
    # and the size of the output layer to number of categories
    layer_sizes[0][0] = 784
    layer_sizes[len(layer_sizes) - 1][1] = 10

    net = add_layers(net, layer_sizes)
    return net


def print_result(pred, true, n_wrong, n_test):
    if n_wrong < 20:
        string = (f'predicted values: \n'
                  f' {pred} \n'
                  f'true values: \n'
                  f'{true} \n'
                  f'number of wrong answers: \n'
                  f'{n_wrong}')
    else:
        string = (f'the first 20 predicted values: \n'
                  f' {pred[:20]} \n'
                  f'the first 20 true values: \n'
                  f'{true[:20]} \n'
                  f'total number of wrong answers: \n'
                  f'{n_wrong} \n'
                  f'ratio of wrong answers: \n'
                  f'{n_wrong/n_test}')
    print(string)


def create_train_and_test_data():
    """Load train and test data from the MNIST database. Reshape data and one-hot
    encode.
    
    Returns:
    x_train: a vector that contains the pixels of each training sample
    y_train: a vector that contains the one-hot encoded classes of each training sample
    x_test: a vector that contains the pixels of each testing sample
    y_test: a vector that contains the one-hot encoded classes of each test sample
    """
    (x_train, y_train), (x_test, y_test) = load_data()
    (x_train, y_train), (x_test, y_test) = reshape(x_train, y_train, x_test, y_test)
    x_train, x_test = normalize_x(x_train, x_test)
    # one hot encode the output data (from one number to a list of 0s and 1s)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def run(net):

    n_test = int(input('How many numbers would you like to predict (int)?: '))

    test_indices = random.sample(range(len(test_x)), n_test)

    # test on the random sample of the test data
    pred_classes = net.predict_multiple(test_x[test_indices])
    true_classes = correct_values(test_y[test_indices])

    wrong_indices_list = wrong_indices(pred_classes, true_classes)
    n_wrong = len(wrong_indices_list)

    print_result(pred_classes, true_classes, n_wrong, n_test)

    if n_wrong != 0:

        choice = None

        try:
            choice = int(
                input('select 1 if you want to draw a sample of the wrongly categorized numbers: '))
        except:
            pass

        if choice == 1:

            if len(wrong_indices_list) > 10:
                wrong_indices_list = wrong_indices_list[:10]
                print('Drawing only the first 10 wrongly categorized numbers')

            wrongly_categorized = [test_indices[i] for i in wrong_indices_list]

            (_, _), (imgs_test_x, y_test) = load_data()

            if len(wrongly_categorized) > 5:
                num_col = 5
                num_row = len(wrongly_categorized) // num_col
            else:
                num_col = len(wrongly_categorized)
                num_row = 1

            fig, axes = plt.subplots(num_row, num_col,
                                     figsize=(1.5*num_col, 2*num_row))

            for index, value in enumerate(wrong_indices_list):
                test_index = test_indices[value]
                if num_row != 1:
                    ax = axes[index//num_col, index % num_col]
                elif num_col != 1:
                    ax = axes[index % num_col]
                else:
                    ax = axes
                ax.imshow(imgs_test_x[test_index], cmap='gray')
                ax.set_title(
                    f'''Correct class: {y_test[test_index]}\n Predicted class: {pred_classes[value]}''', y=1.5)

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = create_train_and_test_data()

    network = create_network()
    network, last_error = train_network(network, train_x, train_y, epochs=15, n_train=2000)

    run(network)
