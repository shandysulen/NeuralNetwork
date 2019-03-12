# 2D Point Classification Neural Network

## Summary

Both `keras_nn.py` and `nn.py` execute vanilla neural networks that classify 2D points under the following condition:

* If |x<sup>(1)</sup>| > 1 and |x<sup>(2)</sup>| < 1, then x in C<sub>1</sub>
* If |x<sup>(1)</sup>| < 1 and |x<sup>(2)</sup>| > 1, then x in C<sub>2</sub>
* Asides from (1,0), (-1,0), (0,1), (0,-1), all other 2D points are not permitted

`nn.py` holds a homegrown neural network, whereas `keras_nn.py` contains a Keras-enabled neural network. A major motivation behind this assignment is to compare the differences between the Keras neural network and my own implementation. Thus, both models were configured to be as similar as possible. The input layer holds 3 neurons (x<sup>(1)</sup>, x<sup>(2)</sup>, and 1 for bias). A minimum of one hidden layer consisting of three units its required to achieve 100% classification accuracy, in both cases. The output layer consists of a single neural unit. If the value of the final activation is approximately 0, the test pattern is predicted to belong to C<sub>1</sub>. Likewise, if the value is approximately 1, the pattern is predicted to belong to C<sub>2</sub>.

(1,0), (-1,0), (0,1), (0,-1), and 96 randomly generated 2D points comprise the dataset to make up a total of 100 training/testing patterns. Although it is guaranteed that there are 50 instances of each class in the dataset, the distribution of individual values themselves is unkown. The 96 data points all satisfy the following rules:

* For all C<sub>1</sub> patterns, x = (x<sup>(1)</sup>, x<sup>(2)</sup>), where 1 < |x<sup>(1)</sup>| &leq; 2 and 0 &leq; |x<sup>(2)</sup>| < 1
* For all C<sub>2</sub> patterns, x = (x<sup>(1)</sup>, x<sup>(2)</sup>), where 0 &leq; |x<sup>(1)</sup>| < 1 and 1 < |x<sup>(2)</sup>| &leq; 2

All data points can be found in the beginning of either `keras_nn.py` or `nn.py`. The code responsible for the generation of these data points is commented out at the beginning of either file. Similarly, the weight matrices (which hold the biases) are initialized to the same values for both neural nets, and the code that generated these weight matrices are found right below the data point generation code.

Both neural networks share these identical properties:

* Same starting weight matrices
* Same datasets (shuffled the same way)
* Number of Hidden Layers: 1 (3 units)
* Gradient Descent Learning Rate: 0.2
* Activation Function: Sigmoid Function (w/ slope = 1)
* Epochs Executed: 10,000

Three training/testing distributions are executed within both neural nets:

1. All 100 patterns used for both training and testing (final accuracy represents training accuracy)
2. 80% training/20% testing (testing patterns taken from the end of the feature matrix)
3. 70% training/30% testing (testing patterns taken from the end of the feature matrix)

Specific results for each case is found below.

## Homegrown Neural Network

The homegrown neural network was modeled after the neural net created within this [BogoToBogo tutorial](https://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php). As was mentioned earlier, a minimum of one hidden layer, holding three units, is required to achieve 100% clasification accuracy, except for the 70%/30% split, where only 96% accuracy was achieved. Standard sigmoid activation functions are used for both the hidden layer and the output layer. Binary cross entropy is used as the loss function. The follow tables show loss, accuracy, the approximate number of epochs needed for convergence in each trial, and the weight matrices of each trial. (Convergence in this case is defined when the change in loss is less than &epsilon; = 0.00001)

![Keras Results](images\My_Results.PNG)
![Keras Weights](images\My_Weights.PNG)

## Keras Neural Network Characteristics & Results

The Keras neural network intializes its weight matrices via the `weights_init_2` and `weights_init_3` functions. Standard sigmoid activation functions are used for both the hidden layer and the output layer. Binary cross entropy is used as the loss function, and stochastic gradient descent (SGD) -- tweaked to a learning rate of 0.2 -- optimizes the model, i.e. SGD finds the local (or global) minimum of the loss function. Batch sizes match the number of training patterns (like in the homegrown neural net) which degrades SGD to regular gradient descent. The follow tables show loss, accuracy, the approximate number of epochs needed for convergence in each trial, and the weight matrices of each trial. (Convergence in this case is defined when the accuracy of the model becomes 100% while training.)

![Keras Results](images\Keras_Results.PNG)
![Keras Weights](images\Keras_Weights.PNG)


## Execution

Running `keras_nn.py` and `nn.py` will fit and predict on the respective models three times with distributions listed above in the Summary section. 

Both `keras_nn.py` and `nn.py` were tested on the Windows Python 3.7.2 interpreter. It is unsure how the programs will perform on different version interpreters or on different operating systems.