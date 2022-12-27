# Testing document

The program is tested with unittests. In addition, the effect of the number of layers, epochs and input size is tested by running the program with different parameters. These tests can be run by running time_complexity_tests.py

## Unit testing

### TestLayer class
- Tests the functionalities of each layer using a StubLayer -class to eliminate the issues that randomness in parameters would pose for testing. Tests all methods of the class, mainly covering the following test cases:
  - the sizes of the weights and biases -matrices are correct
  - the output of forward propagation is within the correct range at all layers
  - the output of the layer is different from the input in forward propagation.
  - the parameters are adjusted correctly in backward propagation.

### TestNetwork class
- Tests the functionalities of the entire network using a StubLayer -class for the layers of the network. Tests all methods of the class, mainly covering the following test cases:
  - data flows from one layer to another
  - the output of the last layer is of correct shape and the numbers are within the specified range
  - the weights and biases of each layer change in backward propagation

### TestActivation class
- Tests that the output of the activation functions is in the correct range and is of correct shape.

### TestIndex class
- Tests the index_mnist.py() module covering the following tests cases:
  - the loaded train and test data is of correct shape and type.
  - normalizing the train and test data yields values that are within the correct range.
  - creating a network truly creates the correct type of network.
  - the error value decreases at each epoch and that finally, the epoch error is smaller than 5%.

### TestError class
- Tests that the output of the error function is correct. Note that the error is only for displaying purposes and doesn't affect the program.

## Testing the effect of different parameters

All of these results have been obtained by running time_complexity_tests.py.

### Effect of the number of layers on the running time and the error

The effect of increasing the number of layers has been tested by creating a network with the number of layers ranging from 1 to 10 and running a sample of 2000 and 15 epochs through the network in the training phase. After each training phase the network is tested with a test sample of 100 numbers. Interestingly, the errors increase when more layers are added. I haven't been able to identify the reason for this behavior. However, I would assume that because the correct error is calculated at the last layer and then propagated further, the more layers there are, the further the error at each layer is from the correct error. Hence, the parameters are adjusted using wrong gradient descent and the error increases instead of decreasing.

Another thing to notice here is that adding too many layers leads to overfitting as the last epoch error (=error in the test set) is significantly lower than the error on the test set. There is also a significant jump in the test error when adding the seventh layer to the network. After that phase, the program performs almost as badly as randomly guessing the correct class.

When it comes to running time, adding more layers increases it as one would guess, because the same calculations are performed at each layer. However, adding layers does not yield a linear increase in running time as would be expected because each layer is similar to the previous one (100x100) and hence, should yield 10 000 new calculations. However, what we notice here is that the running time increases more when adding the 8th and 9th layer but then decreases when adding thr 10th layer. I haven't figured out a reasonable explanation for this behavior either.

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/layer_size_graph.png)

### Effect of the number of epochs on the running time and the error

![alt text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/epoch_number_graph.png)

### Time complexity tests

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/sample_size_graph.png)
