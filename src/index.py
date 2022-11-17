
from neural_network.network import Network
from neural_network.layer import Layer
import numpy as np

net = Network()
net.add_layer(Layer(3,3))
net.add_layer(Layer(3, 3))

# currently the network has the following structure
# input layer of two neurons
# hidden layer of two neurons
# output layer of two neurons

#output = net.forward_propagation([1,1, 1])

output = net.forward_propagation(np.array([1,1, 1]))

print(output)