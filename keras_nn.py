import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

X = np.array([[ 1.        ,  0.        ],
              [-1.        ,  0.        ],
              [ 0.        ,  1.        ],
              [ 0.        , -1.        ],
              [ 1.86004048,  0.81039573],
              [ 1.39548481,  0.57975754],
              [ 1.86765556,  0.02318782],
              [ 1.86753575,  0.61572833],
              [ 1.48815964,  0.08130889],
              [ 1.40336706,  0.37551807],
              [ 1.0157143,   0.81919393],
              [ 1.2737145,   0.02703558],
              [ 1.25775937,  0.97698357],
              [ 1.14590216,  0.86401527],
              [ 1.96112098,  0.35521333],
              [ 1.43076311,  0.82921215],
              [ 1.12962297,  0.65749058],
              [ 1.3324335,   0.20077048],
              [ 1.40904487,  0.44868116],
              [ 1.79181838,  0.05871277],
              [ 1.95355251,  0.11136214],
              [ 1.41482577,  0.63816487],
              [ 1.59368282,  0.05234295],
              [ 1.55307629,  0.39626945],
              [ 1.98376748,  0.23544166],
              [ 1.35338063,  0.50114844],
              [ 1.52875359,  0.6753622 ],
              [ 1.50008732,  0.40768953],
              [ 1.60401554,  0.53139392],
              [ 1.97765449,  0.57331577],
              [ 1.75093087,  0.60475455],
              [ 1.30796486,  0.93540906],
              [ 1.96124325,  0.67884341],
              [ 1.98806053,  0.58549858],
              [ 1.28779182,  0.369467  ],
              [ 1.31369976,  0.25692171],
              [ 1.85373833,  0.89463783],
              [ 1.91090153,  0.68660172],
              [ 1.77141249,  0.9388523 ],
              [ 1.93026831,  0.67042772],
              [ 1.45852689,  0.8730915 ],
              [ 1.19594248,  0.36592802],
              [ 1.20154072,  0.88353944],
              [ 1.25420299,  0.6652704 ],
              [ 1.35943325,  0.41641404],
              [ 1.0549757,   0.29260021],
              [ 1.75585736,  0.46867983],
              [ 1.56581409,  0.06564459],
              [ 1.97465115,  0.56866017],
              [ 1.34232282,  0.42754081],
              [ 1.33275111,  0.55602928],
              [ 1.58275913,  0.41185182],
              [ 0.57590392,  1.34893285],
              [ 0.28298479,  1.05323391],
              [ 0.69863746,  1.46193637],
              [ 0.57353017,  1.63641581],
              [ 0.16545289,  1.12865765],
              [ 0.79063585,  1.54283225],
              [ 0.65625358,  1.79308035],
              [ 0.92280934,  1.3466393 ],
              [ 0.93905383,  1.20637932],
              [ 0.91823649,  1.69011673],
              [ 0.75652842,  1.61243356],
              [ 0.86082291,  1.85015466],
              [ 0.91682095,  1.27548873],
              [ 0.32634402,  1.15427224],
              [ 0.04789334,  1.72755905],
              [ 0.69539965,  1.05227894],
              [ 0.23430774,  1.75595122],
              [ 0.20323691,  1.80545912],
              [ 0.08391423,  1.25699219],
              [ 0.4883687,   1.82369949],
              [ 0.48142631,  1.38809098],
              [ 0.40071157,  1.23898538],
              [ 0.62220507,  1.59166195],
              [ 0.83688104,  1.71171829],
              [ 0.36145163,  1.08895625],
              [ 0.91379271,  1.63091025],
              [ 0.97518781,  1.09903887],
              [ 0.28046107,  1.84814193],
              [ 0.25117587,  1.322839  ],
              [ 0.69707732,  1.91032723],
              [ 0.51316076,  1.34926436],
              [ 0.0193026,   1.31762969],
              [ 0.56149238,  1.27552432],
              [ 0.81490534,  1.22173703],
              [ 0.32514669,  1.66123096],
              [ 0.10008025,  1.22625773],
              [ 0.37862607,  1.18949876],
              [ 0.44752792,  1.63219263],
              [ 0.37564008,  1.12069815],
              [ 0.24611354,  1.63196918],
              [ 0.2849954,   1.59266288],
              [ 0.50564517,  1.69942959],
              [ 0.57096338,  1.18375059],
              [ 0.35242594,  1.89259332],
              [ 0.52282751,  1.3924369 ],
              [ 0.73388867,  1.06595088],
              [ 0.62889525,  1.5090455 ],
              [ 0.55015311,  1.91358421]])

ones = np.atleast_2d(np.ones(X.shape[0]))
X = np.concatenate((ones.T, X), axis=1)

y = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Create model
model = Sequential()
model.add(Dense(units=3, activation='sigmoid', use_bias=True, 
                kernel_initializer='random_uniform', bias_initializer='zeros',
                input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model and output score
model.fit(X, y, epochs=70_000, batch_size=100)
score = model.evaluate(X, y, batch_size=100)
print(score)



