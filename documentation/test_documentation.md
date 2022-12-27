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

### Test coverage

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/coverage.png)

Test coverage is 92%. The command line interface and time complexity tests have been left out of unittests. In addition, I don't test the train_network function which only calls the train -function of the network class nor correct print formatting.

## Testing the effect of different parameters

All of these results have been obtained by running time_complexity_tests.py.

### Effect of the number of layers on the running time and the error

The effect of increasing the number of layers has been tested by creating a network with the number of layers ranging from 1 to 10 and running a sample of 2000 and 15 epochs through the network in the training phase. After each training phase the network is tested with a test sample of 100 numbers. Interestingly, the errors increase when more layers are added. In addition, there is a significant jump in the test error when adding the seventh layer to the network. After that phase, the program performs almost as badly as randomly guessing the correct class.

I haven't been able to identify the reason for this behavior. However, I would assume that because the correct error is calculated at the last layer and then propagated further, the more layers there are, the further the error at each layer is from the correct error. Hence, the parameters are adjusted using wrong gradient descent and the error increases instead of decreasing.

One thing to notice here is that the error in the training dataset doesn't increase as radically as the error in the test dataset. This means that adding too many layers leads to overfitting.

When it comes to running time, adding more layers increases it as one would guess, because the same calculations are performed at each layer. However, adding layers does not yield a linear increase in running time. This would be the expected behavior because each layer is similar to the previous one (100x100) and hence, should yield 10 000 new calculations. However, what we notice here is that the running time increases more when adding the 8th and 9th layer but then decreases when adding thr 10th layer. I haven't figured out a reasonable explanation for this behavior either.

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/layer_size_graph.png)

### Effect of the number of epochs on the running time and the error

These tests are run by using a network with one hidden layer (the best alternative according to the layer tests) and training sample of 1000 numbers.

The results of increasing the number of epochs (how many times the train sample runs through the network) are not very surprising. Running time increases linearly until 25 epochs whichs is expected, because at each epoch, the same calculations are performed. There should be no reason for the running time to not increase between 25 and 35 epochs.

The error in the train data seems to decrease at each epoch, being less than 1% at best. However, the error in the test data doesn't decrease as radically and even increases after 10 epochs. This is likely due to overfitting the somewhat small training dataset. Error in the test dataset is 10% at best.

![alt text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/epoch_number_graph.png)

### Effect of increasing the sample size

These results have been obtained by training a network with one hidder layer using 15 epochs and varying training sample sizes. 

The effects of increasing the sample size are somewhat unexpected. Let's start with the expected results: The running time increases linearly (looks like exponential in the graph due to x axis scale changing). For example, training the network with 10 000 samples takes around 50 seconds while using a sample size of 60 000 takes 300 seconds.

The errors in the train dataset increase when training sample size is increased. This is logical, because the more samples there are, the less the overfitting there should be and hence, the worse the network performs on a big sample. However, increased sample size should lead to smaller errors in the test dataset which doesn't seem to happen here. For some reason, the errors jump up and down. One explanation could be the small test sample size (100 numbers). Because some numbers are easier to predict than others, the composition of the test sample might change the test error quite a bit.

![alt_text](https://github.com/oliviahorjamo/number-recognizer/blob/main/documentation/data/sample_size_graph.png)
