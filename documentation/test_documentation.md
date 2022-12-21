# Testing document

The program is tested with unittests. In addition, the effect of the number of layers, epochs and input size is tested by running the program with different parameters. These tests can be run by running time_complexity_tests.py

## Unit testing

### TestLayer class
- Tests the functionalities of each layer using a StubLayer -class to eliminate the issues that randomness in parameters would pose for testing. Representative test cases have been used for all methods.

### TestNetwork class
- Tests the functionalities of the entire network using a StubLayer -class for the layers of the network. Tests all methods of the class, mainly covering the following test cases:
  - data flows from one layer to another
  - the output of the last layer is of correct shape and the numbers are within the specified range
  - the weights and biases of each layer change in backward propagation

### TestActivation class
- Tests that the output of the activation functions is in the correct range and is of correct shape.

### TestIndex class
- Tests the index_mnist.py() module.
  - Tests that the loaded train and test data is of correct shape and type.
  - Tests that normalizing the train and test data yields values that are within the correct range.
  - Tests that creating a network truly creates the correct type of network.
  - Tests that the error value decreases at each epoch and that finally, the epoch error is smaller than 5%.

## Testing the effect of different parameters

### Effect of the number of layers on the running time and the error

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/layer_size_graph.png)

### Effect of the number of epochs on the running time and the error

![alt text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/epoch_number_graph.png)

### Time complexity tests

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/sample_size_graph.png)
