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

    def train(self, training_samples, epochs, learning_rate):
        """train the network with the training data
        
        Parameters:
        training_samples: tuples with the input data and the correct output as numpy arrays
        epochs: the number of training epochs i.e. how many times to run this training data
                through the network
        learning_rate: an integer that telss how much to adjust the weigths in each update"""
        
        n_samples = len(training_samples)

        # loop through the training data as many times as the epochs -parameter tells
        for i in range(epochs):
            epoch_error = 0
            sample = 0
            for j in training_samples:
                sample += 1
                x = j[0]
                y_true = j[1]
                #print('input vector in network train')
                #print(x)
                y_pred = self.forward_propagation(x)
                #print('predicted output')
                #print(y_pred)

                output_error = mse_gradient(y_true, y_pred)

                #print('output error gradient of the last layer')
                #print(output_error)

                # calculate the error value to see if it gets smaller
                error = mean_squared_error(y_true, y_pred)
                epoch_error += error
                #print('error in sample ', sample, ':', error)
                self.backward_propagation(output_error, learning_rate)
            print('average epoch error', epoch_error / n_samples)