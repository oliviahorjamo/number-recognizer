# Testing document

The program is currently tested only with unit tests. In addition, manual testing has been done at all phases of the project.

## Unit testing

### TestLayer class
- Tests the functionalities of each layer using a StubLayer -class to eliminate the issues that randomness in parameters would pose for testing. Representative test cases have been used for all methods.

### TestNetwork class
- Tests the functionalities of the entire network using a StubLayer -class for the layers of the network. Currently focuses on testing that the data flows correctly from one layer to another.
- Later focus should move to testing that the error decreases when the network is trained and that the predictions are sufficiently accurate.

### TestLoss class
- Tests the loss functions

## TestActivation class
- Tests the activation functions
