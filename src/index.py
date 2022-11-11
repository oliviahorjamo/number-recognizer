
from neural_network.network import Network
from neural_network.layer import Layer

net = Network()
net.add_layer(Layer(2,2))
net.add_layer(Layer(2, 2))

# currently the network has the following structure
# input layer of two neurons
# hidden layer of two neurons
# output layer of two neurons

output = net.forward_propagation([1,1])

print(output)