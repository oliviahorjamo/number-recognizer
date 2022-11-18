
from neural_network.network import Network
from neural_network.layer import Layer
import neural_network.loss_functions as loss_functions
import numpy as np

net = Network()
net.add_layer(Layer(3, 3))
net.add_layer(Layer(3, 3))
net.add_layer(Layer(3, 3))

# currently the network has the following structure
# input layer of three neurons
# hidden layer of three neurons
# output layer of three neurons

# this is of wrong shape, must fix!
y_pred = net.forward_propagation(np.array([[0.5, 0.5, 0.5]]))

# this test case would represent classifying into three different classes
y_true = np.array([[0, 0, 1]])

# the gradient of the error with respect to the output
# describes which way the output should move to yield a better approximation
output_error_gradient = loss_functions.mse_gradient(y_true, y_pred)

data_after_back_propagation = net.backward_propagation(output_error_gradient)

print(data_after_back_propagation)
