# Testing document

The program is currently tested only with unit tests. In addition, manual testing has been done at all phases of the project.

## Unit testing

### TestLayer class
- Tests the functionalities of each layer using a StubLayer -class to eliminate the issues that randomness in parameters would pose for testing. Representative test cases have been used for all methods.

### TestNetwork class
- Tests the functionalities of the entire network using a StubLayer -class for the layers of the network. Tests all methods of the class, mainly covering the following test cases:
  - data flows from one layer to another
  - the output of the last layer is of correct shape and the numbers are within the specified range
  - the weights and biases of each layer change in backward propagation

### TestActivation class
- Tests the activation functions

### TestIndex class
- Tests the index_mnist.py() module.
