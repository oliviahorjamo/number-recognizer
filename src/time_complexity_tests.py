import os
import csv
from time import time
from neural_network.network import Network
from neural_network.layer import Layer
import index_mnist as index

# each of the functions should write the results to a csv file that can then be
# graphically inspected


CSV_PATH = './documentation/data/'
LAYER_FILE = 'layer_test_results.csv'
EPOCH_FILE = 'epoch_test_results.csv'
SAMPLE_SIZE_FILE = 'sample_size_test_results.csv'
TIME_COMPLEXITY_FILE = 'time_complexity_test_results.csv'

class TimeComplexityTests():
    """A class for testing the time complexity of the program.
    """

    def __init__(self):
        """Attributes:
        self.network: a list of different networks to test. The only thing that
        can differ between the networks is the number of layers."""
        self.networks = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def create_csv_folder(self):
        if not os.path.isdir(CSV_PATH):
            os.makedirs(CSV_PATH)
            print('luotu csv kansio')

    def write_csv_file(self, filename, data):
        with open(CSV_PATH+filename, 'w', encoding='utf-8') as file:
            writer = csv.writer(file)
            if filename == LAYER_FILE:
                writer.writerow(['layers', 'last_error', 'test_error', 'time'])
            elif filename == EPOCH_FILE:
                writer.writerow(['epochs', 'last_error', 'test_error', 'time'])
            elif filename == SAMPLE_SIZE_FILE:
                writer.writerow(
                    ['sample_size', 'last_error', 'test_error', 'time'])
            for row in data:
                writer.writerow(row)

    def print_networks(self):
        for network in self.networks:
            print('network with ', len(network.layers), ' layers')
            for layer in network.layers:
                print('layer, shape: ', layer.weights.shape)

    def predict_and_return_n_wrong(self, net):
        n_test = 1000
        pred = net.predict_multiple(self.x_test[:n_test])
        correct = index.correct_values(self.y_test[:n_test])
        n_wrong = len(index.wrong_indices(pred, correct))
        return n_wrong / n_test

    def test_effect_of_layers(self):
        """ Test the effect of adding more layers both to the error and to the
        time of training and prediction."""
        n_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        for number in n_layers:
            network = Network()
            # for each network, the first layer is always the input layer
            network.add_layer(Layer(28*28, 100))
            for _ in range(1, number-1):
                network.add_layer(Layer(100, 100))
            network.add_layer(Layer(100, 10))
            self.networks.append(network)
        result = []
        for net in self.networks:
            start = time()
            net, last_error = index.train_network(net, self.x_train, self.y_train)
            end = time()
            p_wrong = self.predict_and_return_n_wrong(net)
            result.append([len(net.layers), last_error, p_wrong, end-start])
        self.write_csv_file(LAYER_FILE, result)

    def test_effect_of_epochs(self):
        """ Test the effect of adding more epochs to both the error and the time
        of training and prediction."""
        # train a similar network with different epoch numbers
        epochs = [1, 5, 10, 15, 20, 25, 30, 35]
        result = []
        for n_epochs in epochs:
            net = self.create_basic_net()
            start = time()
            net, last_error = index.train_network(
                net, self.x_train, self.y_train, epochs=n_epochs)
            end = time()
            p_wrong = self.predict_and_return_n_wrong(net)
            result.append([n_epochs, last_error, p_wrong, end-start])
        self.write_csv_file(EPOCH_FILE, result)

    def test_effect_of_sample_size(self):
        """Test the effect of increasing and decreasing sample size both to
        the error and the time of training and prediction."""
        # train a similar network with different sample sizes
        sample_sizes = [100,500,1000,2000,3000,4000,5000,6000,
                        7000, 8000, 9000, 10000, 12000,
                        14000, 16000, 18000, 20000,
                        25000,30000,40000,50000,60000]
        result = []
        for size in sample_sizes:
            net = self.create_basic_net()
            start = time()
            net, last_error = index.train_network(
                net, self.x_train, self.y_train, n_train=size)
            end = time()
            p_wrong = self.predict_and_return_n_wrong(net)
            result.append([size, last_error, p_wrong, end-start])
        self.write_csv_file(SAMPLE_SIZE_FILE, result)

    def test_time_complexity(self):
        """Change all parameters and test the time of training."""
        # network 1: n layers, l sample size, m neurons in each hidden layer, e epochs
        # network 2: n+1 layers, l+1 sample size, m+1 neurons in each hidden layer, e+1 epochs
        # network 3: n+2 layers, l+2 sample size, m+2 neurons in each hidden layer, e+2 epochs
        # the result should be that the training time increases to the power of 5
        return True

    def create_basic_net(self):
        net = Network()
        net.add_layer(Layer(28*28, 100))
        net.add_layer(Layer(100, 50))
        net.add_layer(Layer(50, 10))
        return net

    def set_test_datasets(self, x_tr, y_tr, x_te, y_te):
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        self.y_test = y_te

if __name__ == '__main__':
    tests = TimeComplexityTests()
    tests.create_csv_folder()
    x_train, y_train, x_test, y_test = index.create_train_and_test_data()
    tests.set_test_datasets(x_train, y_train, x_test, y_test)
    tests.test_effect_of_layers()
    tests.test_effect_of_epochs()
    #tests.test_effect_of_sample_size()
