# Project specification

The project idea is to create a program to recognize hand written numbers with a neural network that has been created from scratch. Hence, the network should perform a classification task. The exact implemementation of the network will become more accurate later but the network will consist of at least the following:

- A class for the entire network
    - contains multiple different layers
- One class for the base layer of the network
    - Properties: the weights of the edges connecting nodes, the bias of each node
    - Methods: back propagation and forward propagation, activation
- Classes for different types of layers (input layer, hidden layers, output layer)
- Cost functions

## Algorithm and Data Structure design

The big picture of how the program works is the following:
1. We give input data for the input layer of the neural network
2. The data flows from one layer to another until we have the output that will be an array of size (1, number of classes). The value of each class in the output is the probability of the input being of the given class. Each time data flows from one layer to another, an activation function is used to decide which nodes should activate.
3. Once we have the output, we can calculate the error and use it to adjust the parameters of the network (weights and biases).
4. Iterating through this process until the required accuracy is reached.

There will be multiple algorithms needed in this project, mainly related to training the layers. The most important algorithms will be **forward propagation** and **backward propagation**. Forward propagation means that the output of one layer serves as an input for the next layer, i.e. data flows from the input layer to the output layer. Backward propagation, on the other hand, means the process of moving information from the output layer to the input layer. In each layer, an activation function will be used to determine how the input should change.

The neural network will be implemented as multiple matrices that will be stored as numpy arrays. For each layer i and layer i+1, there will be a matrix to represent the weigths on the edges between these layers.

#### Time Complexity

The time complexity of the project depends mainly on the time complexity of the forward and back propagation algorithms and the number of times they must be run.

**The time complexity of forward propagation:**
- When going from layer i to layer j, the output data must be calculated by multiplying the input data by the weights of the edges between the nodes of layer i and layer j. This is performed as a matrix multiplication (O(n^3)).
- Then we will apply an activation function, the time complexity of which I unfortunately haven't had time to figure out yet.
- This is repeated on every layer and every training epoch. Hence, the time complexity of forward propagation will be O(n⁵).

**The time complexity of backward propagation:**
- Unfortunately, I haven't had time to calculate the time complexity of back propagation yet.

#### Space Complexity

We need to store all matrices that represent the weights between the layers. If the number of layers is n and the size of input is m, this will yield space complexity O(n2m).


## Program input and output
The program will get as an input a hand-written number between 0 and 9. This will most likely come from the MNIST database. The program will output a single number representing its prediction of what the input number was.

## Programming and documentation language

In this project, Python is being used as a programming language and English is being used as a documentation language.

## Degree programme

Bachelor's in Computer Science

## Sources

https://www.sciencedirect.com/science/article/pii/S0925231217307555
https://lunalux.io/computational-complexity-of-neural-networks/
https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication
