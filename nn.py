import numpy as np
import time

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - x**2

class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid'):

        # Set nonlinear activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # Set weights and layer config
        self.weights = []
        self.layers = layers
        
        for i in range(1, len(layers) - 1):
            rand_weight_mat = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
            self.weights.append(rand_weight_mat)
    
        rand_weight_mat = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(rand_weight_mat)
    
    def fit(self, X, y, learning_rate=0.2, epochs=100_000):
        # Add column of ones to X (for bias)
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            print(i)
            a = [X[i]]
            print(a)

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])                
                activation = self.activation(dot_value)                
                a.append(activation)
            
            print("A")
            print(a)

            # output layer
            error = y[i] - a[-1]
            print(f"Error: {error}")
            deltas = [error * self.activation_deriv(a[-1])]

            # Begin at second-to-last layer             
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()
            print(f"Deltas: {deltas}")

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                dot_product = layer.T.dot(delta)
                print("Dot Product:")
                print(dot_product)
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print(f'epochs: {k}' )

    def predict(self, x): 
        a = np.hstack((np.ones(1), np.array(x)))   
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return 0 if a <= 0.5 else 1

if __name__ == "__main__":
    nn = NeuralNetwork([2,2,1], 'tanh')
    print("Weights from Layer 1 to Layer 2: \n")
    print(nn.weights[0])

    print("\nWeights from Layer 2 to Layer 3: \n")
    print(nn.weights[1])

    # Initialize training set matrix X and target vector y
    X = np.array([[1,0],
                  [-1,0],
                  [0,1],
                  [0,-1]])
    
    y = np.array([0, 0, 1, 1])

    # Add 48 extra class 1 values
    for i in range(48):
        x1 = abs(np.random.random() - 1) + 1 # (1,2]
        x2 = np.random.random() # [0,1)        
        X = np.concatenate((X, np.array([x1, x2]).reshape((1,2))), axis=0)
        y = np.append(y, 0) 

    # Add 48 extra class 2 values
    for i in range(48):
        x1 = np.random.random() # [0,1)        
        x2 = abs(np.random.random() - 1) + 1 # (1,2]
        X = np.concatenate((X, np.array([x1, x2]).reshape((1,2))), axis=0)
        y = np.append(y, 1) 

    # Review X and y
    # print("\nX:\n")
    # print(X)
    # print(X.shape)

    # print("\ny:\n")
    # print(y)
    # print(y.shape)
    start_time = time.time()
    nn.fit(X,y)
    print(f"Time Elapsed (Model Fitting): {time.time() - start_time} s")
    print(f"Final weights:")
    print(nn.weights)
    c = 0
    for i in range(X.shape[0]):
        if nn.predict(X[i]) == y[i]: c += 1
    
    print(f"Accuracy: {(c / len(y)) * 100}%")





    



