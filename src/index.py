
from neural_network.network import Network
from neural_network.layer import Layer
import neural_network.loss_functions as loss_functions
import numpy as np

#at this point, test with XOR task

#x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])

# the first neuron means class 0 and the second neuron means class 1
#y_train = np.array([[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]])

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

#train_data = [
#            (np.array([[0,0]]), np.array([1])),
#            (np.array([[0,1]]), np.array([0])),
#            (np.array([[1,0]]), np.array([0])),
#            (np.array([[1,1]]), np.array([1]))
#            ]

#train_data = [
 #           (np.array([[0,0]]), np.array([[0, 1]])),
#]

#print(x_train[0])
#print(train_data[0][0])

#x_train = np.array([[0.3, 1],
 #           [0.5, 0.2],
  #          [1, 0.4]])

#y_train = np.array([[0.75],
 #                   [0.82],
  #                  [0.93]])

print(x_train)
print(y_train)

net = Network()

input_size = np.size(x_train[0])
output_size = np.size(y_train[0])

print(input_size)
print(output_size)

net.add_layer(Layer(input_size, 3))
#net.add_layer(Layer(2, 2))
net.add_layer(Layer(3, output_size))

y_pred = net.predict(x_train)

print('y_pred', y_pred)

net.train(x_train, y_train, 1000, learning_rate = 0.1)
#net.train(train_data, 1, learning_rate = 0.1)


#test_data = np.array([[1,0]])
y_pred = net.predict(x_train)
#print('test data:', test_data)

print('y_pred', y_pred)