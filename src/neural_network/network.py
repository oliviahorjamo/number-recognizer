import numpy as np
from neural_network.loss_functions import mean_squared_error


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
            output_error = layer.backward_propagation(
                output_error, learning_rate)
        return output_error

    def predict_multiple(self, test_array):
        """predict the class of multiple input arrays and
            return a list of the predicted values"""
        predictions = []
        for array in test_array:
            class_label = self.predict(array)
            predictions.append(class_label)
        return predictions

    def predict(self, input_array):
        """predict the class of the input array"""
        predicted_array = self.forward_propagation(input_array)
        max_class = np.argmax(predicted_array)
        return max_class + 1

    def train_epoch(self, x_train, y_train, learning_rate):
        """Loop through all samples and call the training function for each
        sample.

        Parameters:
        x_train: x values for the n training samples
        (n can be set at index_mnist.py train -function)
        y_train: the true values for the training samples
        learning rate: a float value that tells how radically the weights and biases
        should be changed
        """
        n_samples = len(x_train)
        epoch_error = 0
        for i in range(n_samples):

            epoch_error += self.train_sample(
                x_train[i], y_true=y_train[i], learning_rate=learning_rate)

        return epoch_error / n_samples

    def train_sample(self, sample, y_true, learning_rate):
        """flow data through one sample and adjust weights and biases

        Parameters:
        x: the training sample
        y_true: the true value for this sample
        learning_rate: a float value that tells how radically the weights
        and biases should be changed
        """
        y_pred = self.forward_propagation(sample)
        error_value = mean_squared_error(y_true, y_pred)
        error = - (y_true - y_pred)
        self.backward_propagation(error, learning_rate)
        return error_value

    def train(self, x_train, y_train, epochs, learning_rate):
        """train the network with the training data

        Parameters:
        training_samples: tuples with the input data and the correct output as numpy arrays
        epochs: the number of training epochs i.e. how many times to run this training data
                through the network
        learning_rate: an integer that telss how much to adjust the weigths in each update"""

        # loop through the number of epochs
        for i in range(epochs):
            # call the function that trains each epoch
            epoch_error = self.train_epoch(x_train, y_train, learning_rate)
            print(f'epoch {i+1}/{epochs}   error={epoch_error}')
        return epoch_error
