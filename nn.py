import numpy as np
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
import time

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - x**2

def get_loss(X,y):
    loss = 0
    for i in range(len(y)):
        loss += binary_crossentropy(X[i], y[i])

def get_accuracy(nn, X, y):
    c = 0
    for i in range(X.shape[0]):
        if nn.predict(X[i]) == y[i]: c += 1
    return c / len(y)

def weights_init_2():
    return np.array([[ 0.89281447,  0.1173421,   0.40226235],
                     [-0.99602829,  0.34000123,  0.85147312],
                     [-0.03140238, -0.5338699,  -0.13181507]])

def weights_init_3():
    return np.array([[-0.97794294],
                     [-0.86163768],
                     [ 0.49360355]])

# """
# h: hypothesis value
# t: target value
# """
# def get_binary_cross_ent_deriv(h, h_deriv, t):
#     return -(t * h_deriv / h) + ((1 - t) * h_deriv / (1 - h))

"""
h: hypothesis value
t: target value
"""
def get_binary_cross_ent_loss(h, t):
    if h == 0: h = 0.0000001
    return -np.log(h) if t == 1 else -np.log(1-h)

"""
a: vector of output activations
y: target vector
"""
def get_total_loss(a, y):
    loss = 0
    for i in range(len(y)):
        loss += get_binary_cross_ent_loss(a[i], y[i])
    return loss

        
# # Initialize training set matrix X and target vector y
# X = np.array([[1,0],
#               [-1,0],
#               [0,1],
#               [0,-1]])

# y = np.array([0, 0, 1, 1])

# # Add 48 extra class 1 values
# for i in range(48):
#     x1 = abs(np.random.random() - 1) + 1 # (1,2]
#     if np.random.random() < 0.5: x1 *= -1
#     x2 = np.random.random() # [0,1)
#     if np.random.random() < 0.5: x2 *= -1 
#     X = np.concatenate((X, np.array([x1, x2]).reshape((1,2))), axis=0)
#     y = np.append(y, 0) 

# # Add 48 extra class 2 values
# for i in range(48):
#     x1 = np.random.random() # [0,1)
#     if np.random.random() < 0.5: x1 *= -1
#     x2 = abs(np.random.random() - 1) + 1 # (1,2]
#     if np.random.random() < 0.5: x2 *= -1
#     X = np.concatenate((X, np.array([x1, x2]).reshape((1,2))), axis=0)
#     y = np.append(y, 1)

# Generate weight matrices to map from layer 1 to layer 2 (3x3) and from layer 2 to layer 1 (3x1)
# Weight matrices contain the bias 
# for i in range(1, len(layers) - 1):
#     rand_weight_mat = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
#     self.weights.append(rand_weight_mat)

# rand_weight_mat = 2 * np.random.random((layers[i] + 1, layers[i+1])) - 1
# self.weights.append(rand_weight_mat)

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
        self.weights = [weights_init_2(), weights_init_3()]
        self.layers = layers        
    
    def fit(self, X, y, learning_rate=0.2, epochs=10_000):
        # Add column of ones to X (for bias)
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs+1):
            total_a = []

            for i in range(len(y)):          
                a = [X[i]]                 

                # Forward propagation to record activations
                for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])                       
                    activation = self.activation(dot_value)                               
                    a.append(activation)
                
                # Record last activation (output) for total loss
                total_a.append(a[-1][0])

                # Backward propagation to record deltas
                error = y[i] - a[-1]            
                deltas = [error * self.activation_deriv(a[-1])]

                # Begin at second-to-last layer             
                for l in range(len(a) - 2, 0, -1): 
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                                
                deltas.reverse() # [level2(hidden)->level3(output)]

                # Weight update
                for j in range(len(self.weights)):
                    layer = np.atleast_2d(a[j])
                    delta = np.atleast_2d(deltas[j])
                    dot_product = layer.T.dot(delta)
                    self.weights[j] += learning_rate * layer.T.dot(delta)

            if k % 1000 == 0: print(f"epochs: {k}, loss: {get_total_loss(total_a, y)}")

    def predict(self, x): 
        a = np.hstack((np.ones(1), np.array(x)))        
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))                 
        return 0 if a <= 0.5 else 1

if __name__ == "__main__":

    # Initialize data
    X = np.array([[ 0.51379800, -1.97092809],
                  [-0.20984328, -1.23154396],
                  [-1.96127284,  0.25484967],
                  [-1.11089456,  0.74969683],
                  [ 0.14293785, -1.67208987],
                  [-1.90582219,  0.98317882],
                  [-1.21887848,  0.65923364],
                  [ 0.93263703, -1.96404054],
                  [ 1.21915466, -0.72743944],
                  [-1.75319409,  0.91845219],
                  [-1.46278318,  0.15180805],
                  [-0.67408359, -1.07303636],
                  [-1.35583225, -0.32737513],
                  [-0.30674159,  1.69562315],
                  [-0.14992217, -1.77398921],
                  [ 0.99766389,  1.78601844],
                  [-0.35050087,  1.15534855],
                  [ 1.58196262, -0.05505483],
                  [ 0.55385997,  1.64149097],
                  [-0.08949460,  1.50214932],
                  [ 1.85693801,  0.98301599],
                  [-1.95422168, -0.99030333],
                  [-0.44891320, -1.11369429],
                  [-1.57236923, -0.64347211],
                  [ 0.86665464, -1.32719849],
                  [ 1.04463436,  0.37440069],
                  [ 1.33379971,  0.66699667],
                  [ 1.50541199, -0.34104633],
                  [ 0.66609976, -1.76033983],
                  [ 1.87854077,  0.44098273],
                  [-0.00801912,  1.31453792],
                  [-0.04114719, -1.51247422],
                  [ 0.55702237,  1.4678183 ],
                  [ 1.70887396, -0.98088073],
                  [ 0.11786665, -1.81535059],
                  [ 1.92533888,  0.60888706],
                  [ 0.00000000, -1.        ],
                  [ 0.03712645, -1.47039334],
                  [ 1.21375786,  0.57652999],
                  [-1.62441754, -0.04454102],
                  [-1.98113875, -0.57797333],
                  [-0.71951219, -1.10508211],
                  [ 1.12344714,  0.13266985],
                  [-0.71753223,  1.59492958],
                  [-0.73441611, -1.27312027],
                  [ 1.15364458,  0.67051151],
                  [-0.03668461,  1.36274862],
                  [ 0.83768198,  1.00285609],
                  [-0.43684742,  1.09788747],
                  [ 1.00000000,  0.        ],
                  [ 0.65554653,  1.70522318],
                  [ 0.00000000,  1.        ],
                  [-0.43454150,  1.97784049],
                  [-1.23363963,  0.21983501],
                  [ 1.86986835, -0.63102375],
                  [-1.63202492, -0.61562931],
                  [ 0.92768524, -1.07770172],
                  [-0.77921779,  1.88984142],
                  [ 0.77792292,  1.728878  ],
                  [ 1.24661371, -0.38538536],
                  [ 0.00758863,  1.61173528],
                  [ 1.26564181,  0.01916255],
                  [ 1.13414766,  0.11282361],
                  [ 1.56623998,  0.01078686],
                  [-1.00000000,  0.        ],
                  [-0.06683438,  1.21864147],
                  [ 1.13897306, -0.19234917],
                  [-1.76069733, -0.02390209],
                  [ 1.74694586, -0.13678508],
                  [-0.04899890, -1.12102474],
                  [-1.54964574,  0.72268797],
                  [ 0.42145793, -1.34575556],
                  [-1.93803846, -0.46829894],
                  [ 1.22533514,  0.7740759 ],
                  [-0.04449343,  1.78868134],
                  [ 0.35421534, -1.94829106],
                  [ 0.85747417,  1.66593297],
                  [ 0.92333213, -1.9160327 ],
                  [ 1.12643162, -0.73849726],
                  [-1.17467587, -0.1531918 ],
                  [ 0.61721263, -1.32722085],
                  [-0.93272475,  1.22761548],
                  [-0.40201952, -1.8442309 ],
                  [-1.45407790, -0.84518787],
                  [-0.43879207,  1.93351689],
                  [-1.87223659, -0.45155528],
                  [ 1.83738101, -0.83005961],
                  [ 1.61038303,  0.54139823],
                  [ 1.73139207,  0.99539367],
                  [-1.89530746, -0.62162705],
                  [-0.52246682, -1.70648464],
                  [ 0.90959735, -1.85112022],
                  [-0.18730388,  1.87750163],
                  [ 1.81073439,  0.91240518],
                  [-1.30205126,  0.70281808],
                  [ 1.07985431, -0.66470045],
                  [-0.72190509,  1.09964043],
                  [ 1.72962297,  0.55908371],
                  [-0.08897268,  1.44571676],
                  [ 0.37671337, -1.07039618]])

    y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1,
 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1])

    # Create neural network
    nn = NeuralNetwork([2,2,1], 'sigmoid')

    # 1. All 100 patterns used for both training and testing (final accuracy represents training accuracy)
    print("1. All 100 patterns used for both training and testing (final accuracy represents training accuracy)")
    start_time = time.time()
    nn.fit(X, y, epochs=10_000)
    print(f"Time Elapsed (Model Fitting): {time.time() - start_time} s")
    
    # Get accuracy of model
    accuracy = get_accuracy(nn, X, y)
    print(f"Accuracy: {accuracy * 100}%")

    # Show final weight values
    print(f"\nFinal Layer 1 Weight Matrix:")
    print(nn.weights[0])
    print(f"\nFinal Layer 2 Weight Matrix:")
    print(nn.weights[1])

    # 2. 80% training/20% testing (testing patterns taken from the end of the feature matrix)
    nn = NeuralNetwork([2,2,1], 'sigmoid')

    print("\n\n2. 80% training/20% testing (testing patterns taken from the end of the feature matrix)")
    start_time = time.time()
    (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.2)
    nn.fit(X_train, y_train, epochs=10_000)
    print(f"Time Elapsed (Model Fitting): {time.time() - start_time} s")
    
    # Get accuracy of model
    accuracy = get_accuracy(nn, X_test, y_test)
    print(f"Accuracy: {accuracy * 100}%")

    # Show final weight values
    print(f"\nFinal Layer 1 Weight Matrix:")
    print(nn.weights[0])
    print(f"\nFinal Layer 2 Weight Matrix:")
    print(nn.weights[1])

    # 3. 70% training/30% testing (testing patterns taken from the end of the feature matrix)
    nn = NeuralNetwork([2,2,1], 'sigmoid')

    print("\n\n3. 70% training/30% testing (testing patterns taken from the end of the feature matrix)")
    start_time = time.time()
    (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.3)
    nn.fit(X_train, y_train, epochs=10_000)
    print(f"Time Elapsed (Model Fitting): {time.time() - start_time} s")
    
    # Get accuracy of model
    accuracy = get_accuracy(nn, X_test, y_test)
    print(f"Accuracy: {accuracy * 100}%")

    # Show final weight values
    print(f"\nFinal Layer 1 Weight Matrix:")
    print(nn.weights[0])
    print(f"\nFinal Layer 2 Weight Matrix:")
    print(nn.weights[1])
