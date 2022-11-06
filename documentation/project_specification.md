# Project specification

The project idea is to create a program to recognize hand written numbers with a neural network that has been created from scratch. The type of the neural network that I will create is a Multilayer Perceptron (MLP). The end goal is to be able to give the network a picture of a hand written number and the network should classify it correctly.

## The general structure of a MLP neural network:

- $L$ layers
- each with $N_L$ neurons
- links between each neuron in layer $l$ and each neuron in layer $l-1$
    - each link has a weight $w_ij^l$ which represents the weight between neuron $j$ in layer $l-1$ and neuron $i$ in layer $l$. The weigth describes how much a change in the value of neuron $j$ in layer $l-1$ should affect the value of neuron $i$ in layer $l$. In other words, the weight describes how connected the two neurons are.
    - in addition, each neuron has a bias term that describes how far the prediction is from the desired value

## The structure of the network as matrices

- For each layer, there should be a matrix to represent the weights of the links between this layer and the next layer. For each layer, the size of the matrix fill be number of neurons in this layer * number of neurons in the next layer
- For each layer, there should also be a vector *$b^l$* describing the bias of each neuron of the layer $l$. In other words, $b_i^l$ describes the bias of the $i$th neuron of layer $l$.
- All in all, if there are $L$ layers, there should be 2*$L$ matrices, since there are two matrices per layer. 
- Each matrix is represented as a numpy array in the program

## The structure of the network as classes in the program

The program should contain the following classes to describe the entire network:

- A basic layer class:
    - properties: the two matrices described above
    - methods: forward propagation and backward propagation and different functions needed in these algorithms (will be described later)
- A class for the entire network
    - properties: layers
    - methods: training the model, predicting the outcome

## The big picture of how the program works

1. We give input data for the input layer of the neural network. The input data is a matrix of size 28 * 28 (since there are 28 pixels in each row and column. This will be transformed into a vector of size (1, 748).
3. The data flows from one layer to another until we have the output that will be an array of size (1, number of classes). The value of each class in the output is the probability of the input being of the given class. Each time data flows from one layer to another, an activation function is used to decide which nodes should activate.
4. Once we have the output, we can calculate the error and use it to adjust the parameters of the network (weights and biases).
5. Iterating through this process until the required accuracy is reached.

### Forward propagation: the flowing of data between layers:

Forward propagation means the process of data flowing from the first layer to the last layer. In each layer, the value of each neuron is decided by the following procedure:

1. Multiply the value of each neuron $j$ in the previous layer $l-1$ by the weight of the link between this neuron $i$ ($w_ij^l$). Calculate the sum of these.
2. Apply an activation function to the sum. The activation function is used to 'squeeze' the value to the range [0,1].
3. Do the same procedure again in the next layer.

### Backward propagation: adjusting the weights and biases to minimize the error

Backward propagation means the procedure of data flowing from the last layer to the first layer while adjusting the weights and biases of each layer so that the error term is minimized.

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
