
import numpy as np

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

# load MNIST data from keras datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
# normalize the input data
x_train /= 255
# one hot encode the output
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

net = Network()
# the first layer must have the same number of neurons as there are pixels
net.add_layer(Layer(28*28, 100))               
net.add_layer(Layer(100, 50))
# the last layer must have the same number of neurons as there are classes
net.add_layer(Layer(50, 10))

# train on a subset of the training data
net.train(x_train[0:1000], y_train[0:1000], epochs=30, learning_rate=0.1)

# test on 50 samples
pred_classes = net.predict_multiple(x_test[0:50])
true_classes = correct_values(y_test[0:50])
print("\n")
print("predicted values : ")
print(pred_classes)
print("true values : ")
print(true_classes)
print('number of wrong answers')
print(number_of_wrong_answers(pred_classes, true_classes))