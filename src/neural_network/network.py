#from neural_network.layer import Layer

from neural_network.loss_functions import mse_gradient, mean_squared_error

class Network:
    """a class to represent the entire network
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, input_array):
        """propagate data forward in the network"""
        for layer in self.layers:
            input_array = layer.forward_propagation(input_array)
        return input_array

    def backward_propagation(self, output_error, learning_rate):
        """propagate data backwards in the network
        
        Note that the for loop must go through the layers in reversed
        order since the data is flowing from the last layer to the first layer."""
        for layer in reversed(self.layers):
            output_error = layer.backward_propagation(output_error, learning_rate)
        #print('output error in network back prop')
        #print(output_error)
        return output_error
        
    def predict(self, input_array):
        """predict the class of the input array"""
        return self.forward_propagation(input_array)

    def train(self, x_train, y_train, epochs, learning_rate):

        """train the network with the training data
        
        Parameters:
        training_samples: tuples with the input data and the correct output as numpy arrays
        epochs: the number of training epochs i.e. how many times to run this training data
                through the network
        learning_rate: an integer that telss how much to adjust the weigths in each update"""
        
        n_samples = len(x_train)

        # loop through the training data as many times as the epochs -parameter tells
        for i in range(epochs):
            epoch_error = 0
            for j in range(n_samples):
                # forward propagation
                x = x_train[j]
                y_true = y_train[j]
                y_pred = self.forward_propagation(x)

                # compute loss (for display purpose only)
                epoch_error += mean_squared_error(y_train[j], y_pred)

                # backward propagation
                #error = mse_gradient(y_train[j], output)
                error = - (y_true - y_pred)

                self.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            epoch_error /= n_samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, epoch_error))