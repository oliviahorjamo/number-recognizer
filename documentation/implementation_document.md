# Implementation document

The project is a neural network that can recognize hand-written numbers. The network is created from scratch using arrays and matrix algebra.

## Structure of the program

The program contains two main classes:
- Layer: A class for each layer of the network.
   - Calculating the output of the given layer from the given input.
   - Modifying the weights and parameters of the layer based on the error of the layer.
- Network: A class for the entire network that contains multiple layers.
   - Propagating the results of each layer to the next layer and finally yielding a prediction of the class.
   - Propagating the error backwards so that the parameters at each layer can be adjusted. 

In addition, the program contains index_mnist.py -module which loads the train and test datasets, reshapes them and runs the very simple user interface.

## Datastructures used in the program

The only datastructures that the program uses are numpy arrays which are vectors.

For each layer there are four vectors:
- One for the weights between the previous layer and this layer. The size of this vector is m x n where m is the number of nodes in the previous layer and n is the number of nodes in this layer.
- One for the biases of this layer. The size of this vector is 1 x n where n is the number of nodes in this layer.
-  One for the input of this layer. The size of this vector is 1 x m where m is the number of nodes in the previous layer.
-  One for the output of this layer. The size of this vector is 1 x n where n is the number of nodes in this layer.

The network has only datastructure which is a list of Layer objects.

## The algorithms used in the project

The two algorithms used in this project are forward propagation and backward propagation. Next, I'll present a short summary of both algorithms.

### Forward propagation

Forward propagation is used for running a sample through the network both in the training phase and when predicting. It's a very simple algorithm that contains the following phases.
1. A for loop that runs through each layer in the network from the first layer until the last layer.
2. At each layer, perform a dot product of the input array and the weights between this layer and the previous layer. Add biases. Run the results through an activation function.

I'm not 100% sure of the time complexity of the algorithm but I would assume that it is O(m x n + n + n) = O(m x n) where m is the number of nodes at the previous layer and n is the number of nodes at this layer. This is due to the time complexity the matrix multiplication being O(1 x m x n) assuming the simplest possible matrix multiplication algorithm. After this phase we will have a 1 x n vector at which linear operations are performed (adding biases and applying the activation function).

The time complexity of the entire forward propagation can't be expressed in a straightforward manner because the number of nodes at layers is not constant.

### Backward propagation

Backword propagation algorithm is used for modifying the parameters at each layer according to the error of the prediction. It contains the following phases.
1. Calculate the error of the prediction (predicted classes - correct classes).
2. Run through each layer of the network, starting from the last layer. At each layer calculate the gradient of the error with respect to the input of this layer, the weights and the biases. Each gradient tells which direction the parameters should be modified and by how much to obtain the steepest decrease in the error.
3. Modify the weights and biases according to the error of the gradient with respect to these parameters.

The time complexity of the backward propagation algorithm at each layer should be O(n x n + m x n + n x n + n + n) = O(m x n + n x n) where m is the number of nodes at the previous layer and n is the number of nodes at this layer. This is due to there being three matrix multiplications and two linear vector additions in this phase.

Similarly to the forward propagation algorithm, the time complexity of the entire backward propagation algorithm can't be expressed in a straight forward manner due to the number of nodes not being constant at each layer. However if it was constant, the time complexity should be O(s x t x l x (m x n + n x n)) where t is the number of epochs, s is the sample size and l is the number of layers.

## Flaws and points of improvement

The most important points of improvement is the accuracy of prediction. At the moment, the accuracy is somewhere between 75 and 92% depending on the sample size, the number of layers and the test dataset. I don't know exactly how the performance could be enhanced but here are a few propositions:
1. Change the activation function to something different. Currently I use sigmoid because it is easy to implement. However, there is no guarantee that this is the best alternative.
2. Change the sizes of the hidden layers. There is no exact best alternative for the sizes of the layers so using some other layer size than 100 x 100 might lead to better results.

## Sources:
[1] https://ieeexplore.ieee.org/abstract/document/1193152.

My highlighted document: ![Zhang et al.](https://github.com/oliviahorjamo/number-recognizer/blob/main/Zhang100-ch03.pdf)

[2] https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

[3] https://deeplizard.com/learn/video/m0pIlLfpXWE

[4] https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

[5] https://ai.stackexchange.com/questions/13978/why-is-the-derivative-of-the-activation-functions-in-neural-networks-important

[6] https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication
