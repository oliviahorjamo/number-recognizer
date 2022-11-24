
from neural_network.network import Network
from neural_network.layer import Layer
import neural_network.loss_functions as loss_functions
import numpy as np

#from tests.layer_test import StubLayer

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

train_data = [
            (np.array([[0,0]]), np.array([[0]])),
            (np.array([[0,1]]), np.array([[1]])),
            (np.array([[1,0]]), np.array([[1]])),
            (np.array([[1,1]]), np.array([[0]]))
            ]

net = Network()

input_size = np.size(train_data[0][0])

net.add_layer(Layer(input_size, 2))
#net.add_layer(Layer(2, 2))
net.add_layer(Layer(2, 1))

net.train(train_data, 1000, learning_rate = 0.1)

test_data = (np.array([[1,0]]))
y_pred = net.predict(test_data)
print('test data:', test_data)

print('y_pred', y_pred)