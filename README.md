# 2D Point Classification Neural Network

Both `keras_nn.py` and `nn.py` execute vanilla neural networks that classify 2D points under the following condition:

* If |x<sup>(1)</sup>| > 1 and |x<sup>(2)</sup>| < 1, then x in C<sub>1</sub>
* If |x<sup>(1)</sup>| < 1 and |x<sup>(2)</sup>| > 1, then x in C<sub>2</sub>
* All other 2D points are not permitted

`nn.py` holds a homegrown neural network, whereas `keras_nn.py` contains a Keras-enabled neural network. Both models share similar characteristics. The input layer holds 3 neurons (x<sup>(1)</sup>, x<sup>(2)</sup>, and 1 for bias). The only hidden layer includes 3 units. The output layer consists of a single neural unit. If the value of the final activation is approximately 0, the test pattern is predicted to belong to C<sub>1</sub>. Likewise, if the value is approximately 1, the pattern is predicted to belong to C<sub>2</sub>. Both neural networks perform with 100% accuracy.