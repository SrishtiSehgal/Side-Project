from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
print(network)
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
print(network)
network = max_pool_2d(network, 2)
print(network)
network = local_response_normalization(network)
print(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
print(network)
network = max_pool_2d(network, 2)
print(network)
network = local_response_normalization(network)
print(network)
network = fully_connected(network, 128, activation='tanh')
print(network)
network = dropout(network, 0.8)
print(network)
network = fully_connected(network, 256, activation='tanh')
print(network)
network = dropout(network, 0.8)
print(network)
network = fully_connected(network, 10, activation='softmax')
print(network)
network = regression(network, optimizer='adam', learning_rate=0.01,loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20, validation_set=({'input': testX}, {'target': testY}), snapshot_step=100, show_metric=True, run_id='convnet_mnist')