# Project specification

The project idea is to create a program to recognize hand written numbers with a neural network that has been created from scratch. The type of the neural network that I will create is a Multilayer Perceptron (MLP). The end goal is to be able to give the network a picture of a hand written number and the network should classify it correctly in a class 1...9.

## The general structure of a MLP neural network:

- $L$ layers
- each with $N_L$ neurons
- links between each neuron in layer $l$ and each neuron in layer $l-1$
    - each link has a weight $w_ij^l$ which represents the weight between neuron $j$ in layer $l-1$ and neuron $i$ in layer $l$. The weigth describes how much a change in the value of neuron $j$ in layer $l-1$ should affect the value of neuron $i$ in layer $l$. In other words, the weight describes how connected the two neurons are.
    - in addition, each neuron has a bias term that describes how far the prediction is from the desired value

## The structure of the network as matrices

- For each layer, there should be a matrix to represent the weights of the links between this layer and the next layer. Each column represents the weights between all neurons of the previous layer and one neuron of this layer. Hence, the size of the entire matrix for layer $l$ will be $N_l * N_l+1$ (the number of neurons in this layer * number of neurons in the next layer).
- For each layer, there should also be a vector **$b^l$** describing the bias of each neuron of the layer $l$. In other words, value $b_i^l$ in the vector describes the bias of the $i$th neuron of layer $l$.
- All in all, if there are $L$ layers, there should be 2 * $L$ matrices, since there are two matrices per layer. 
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

1. Input data is given to the input layer of the neural network. The input data is a matrix of size 28 * 28 (since there are 28 pixels in each row and column. This will be transformed into a vector of size (1, 748).
3. The data flows from one layer to another until the output layer has the output that will be an array of size (1, 9). The value of each class in the output is the probability of the input being of the given class. Each time data flows from one layer to another, an activation function is used to decide which nodes should activate.
4. Once we have the output, we can calculate the error and use it to adjust the parameters of the network (weights and biases).
5. Iterating through this process until the required accuracy is reached.

### Forward propagation: the flowing of data between layers:

Forward propagation means the process of data flowing from the first layer to the last layer. In each layer, the value of each neuron is decided by the following procedure:

1. Multiply the value of each neuron $j$ in the previous layer $l-1$ by the weight of the link between this neuron $i$ ( $w_ij^l$ ) and neuron $j$ in the previous layer. Calculate the sum of these values for each neuron.
2. Apply an activation function to each sum and you will get $ x_i^l $, the value of neuron $i$ in layer $l$. The activation function is used to 'squeeze' the value to the range [0,1]. This 
3. Do the same procedure again in the next layer.

**The time complexity of forward propagation:**
- When going from layer $l-1$ to layer $l$, the output value of each neuron is the dot product of the input data vector and the column of this neuron of the weight matrix of this layer. To obtain the entire output data vector, one must compute the matrix multiplication of the input data and the entire weights matrix. The time complexity of a matrix multiplication is $O(n³)$
- After this step, the entire output vector is passed through the activation function. This is an element-wise operation and hence, has the time complexity $O(n²)$
- These steps are repeated at each layer so the time complexity becomes $O(L*(n³+n²)) = O(L * n³)$
- This is the time complexity of running through the network once. Given $P$ training samples and $t$ training epochs, the time complexity becomes $O(L * P * t * n³)$

### Backward propagation: adjusting the weights and biases to minimize the error

Backward propagation means the procedure of data flowing from the last layer to the first layer while adjusting the weights and biases of each layer so that the error term is minimized. This is done by the following procedure:

1. Calculate the error of the output layer which is given as the squared difference between the predicted classes and the desired classes (1 for the correct class and 0 for others).
2. Calculate the negative gradient of the error term which respect to the weights between the previous layer and this layer. The negative gradient gives the direction of steepest decrease, in other words, which wages to change and by how much to decrease the error.
3. Adjust weights with the negative gradient.
4. Repeat at each layer. 


**The time complexity of backward propagation:**
- Unfortunately, I haven't had time to calculate the time complexity of back propagation yet.

#### Space Complexity

We need to store all matrices that represent the weights between the layers. If the number of layers is $N_L$ and the size of input is $m$, this will yield space complexity $O(N_L * m²)$.


## Program input and output
The program will get as an input a hand-written number between 0 and 9. This will come from the MNIST database. The program will output a single number representing its prediction of what the input number was.

## Programming and documentation language

In this project, Python is being used as a programming language and English is being used as a documentation language. I don't know other programming languages sufficiently well to give feedback on projects written in them.

## Degree programme

Bachelor's in Computer Science

## Sources

https://www.sciencedirect.com/science/article/pii/S0925231217307555
https://lunalux.io/computational-complexity-of-neural-networks/
https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication
https://deepai.org/machine-learning-glossary-and-terms/weight-artificial-neural-network
