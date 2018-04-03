Problem statament: Digit recognition using Neural Network

The NN algorithm taking data form ex4data1.mat where the X is the input and Y is the labeled data. Most of the important implementation are preset in utitliy_module_NN.py. 

***********************Important note for using NN************************
1) The inital parameters cannot be 0 reason being if those are zero then each hidden layer will endup computing same function, for this purpose only we have have a symmetry breaking mechanism where we randomly [-epislon,+epislon] we can calculate it as epsilon = sqrt(6)/sqrt(Lin + Lout). We can have some other way as well but as suggested by Andrew NG.

2) After implementing your backpropgation algorithm it is good practice to use gradient checking to confirm that the BP implementaion is correct.

3) Here we are using sigmoid function as output function at each hidden layer and output layer as well, and the function we have to minimize is using: ∑∑(-y(i)*log(h(x))) - ((1-y(i))*log(1-h(x))).

4) A neural network approach is simply propogating the final errors back and updating the relevant weights.

