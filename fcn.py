import numpy as np
import tensorflow as tf
import tflearn
import pandas as pd
import imageio

class NeuralNetBuild:
    def load_data(self):
        X = imageio.imread('original_image.jpg')
        h,w,bpp = np.shape(X)
        X = X.reshape([-1, h,w,bpp])
        self._input_shape = [-1, h, w, bpp]
        print(X.shape)
        Y = imageio.imread('segmented_image.jpg')
        h,w,bpp = np.shape(Y)
        Y = Y.reshape([-1, h, w, bpp])
        print(Y.shape)
        self._signal_train_val_data = [X, Y]
    
    def __init__(self, str_optimizer, f_learning_rate, i_filter_size,i_kernel_size,i_nb_filter,str_activation_function, str_loss_function, i_epochs, f_batch_size): 
        self._optimizer = str_optimizer
        self._learning_rate = f_learning_rate
        self._filter_size = i_filter_size 
        self._kernel_size = i_kernel_size 
        self._nb_filter = i_nb_filter 
        self._loss_function = str_loss_function
        self._activation_function =str_activation_function
        self._batchsize = f_batch_size
        self._epochs = i_epochs
        self._signal_train_val_data = None
        self._input_shape = None
        self.load_data()

def run_model(net):
    def tflearn_conv2d(l_i_shape, str_optimizer, learning_rate, filter_size, kernel_size,nb_filter,activation_function, loss_function,optimizer):
        tflearn.init_graph()
        input_ = tflearn.input_data(shape=[None]+l_i_shape, name='input')
        print(input_)
        layer1conv = tflearn.layers.conv.conv_2d(input_, nb_filter,filter_size, activation=activation_function)#, weight_decay=0.001)
        print(layer1conv)
        layer1pool = tflearn.layers.conv.max_pool_2d(layer1conv, kernel_size)
        print(layer1pool)
        layer4conv = tflearn.layers.conv.conv_2d(layer1pool, nb_filter*2, filter_size, activation=activation_function)#, weight_decay=0.001)
        print(layer4conv)
        layer4pool = tflearn.layers.conv.max_pool_2d(layer4conv, kernel_size)
        print(layer4pool)
        layer6 = tflearn.layers.conv.conv_2d_transpose(layer4pool, nb_filter*2, filter_size,strides = [1,2,2,1], output_shape = layer4conv.get_shape().as_list()[1:])
        print(layer6)
        layer6_skip_connected = tf.math.add(layer6, layer4conv) 
        #layer6_skip_connected = tflearn.layers.merge_ops.merge([layer6, layer4conv], 'sum', axis = 0)
        print(layer6_skip_connected)
        #layer6_skip_connected = tflearn.reshape(layer6_skip_connected, [-1]+layer4conv.get_shape().as_list()[1:])
        layer7 = tflearn.layers.conv.conv_2d_transpose(layer6_skip_connected, nb_filter, filter_size, strides = [1,2,2,1], output_shape =layer1conv.get_shape().as_list()[1:])
        print(layer7) 
        #layer7_skip_connected = tflearn.layers.merge_ops.merge([layer7, layer1conv], 'sum', axis = 0)
        #layer7_skip_connected = tflearn.reshape(layer7_skip_connected, [-1]+layer1conv.get_shape().as_list()[1:])
        layer7_skip_connected = tf.math.add(layer7, layer1conv) 
        print(layer7_skip_connected)
        layer8 = tflearn.layers.conv.conv_2d_transpose(layer7_skip_connected, 3, 1, output_shape =input_.get_shape().as_list()[1:])
        #layer8 = tflearn.reshape(layer8, [-1]+l_i_shape)
        print(layer8)
        net = tflearn.regression(layer8, optimizer= optimizer, loss=loss_function, metric='R2', learning_rate=learning_rate, name='target')
        model = tflearn.DNN(net, tensorboard_verbose=0) # run tensorboard --logdir='/tmp/tflearn_logs'
        return layer4conv, model
    print('in model()\n')
    X, Y = net._signal_train_val_data[0], net._signal_train_val_data[1]
    Y_predicted = None
    with tf.Graph().as_default():
        _, tflearn_model = tflearn_conv2d(net._input_shape[1:], net._optimizer, net._learning_rate, net._filter_size, net._kernel_size,net._nb_filter,net._activation_function, net._loss_function, net._optimizer)
        print('built regressor')
        tflearn_model.fit({'input': X}, {'target': Y},shuffle=True, batch_size=net._batchsize, n_epoch=net._epochs, show_metric=True, run_id ='fcn_image')
        print('fit model')
        tflearn_model.save('fully_conv_net.tflearn')
        print('saving model')
        Y_predicted = tflearn_model.predict(np.atleast_3d(X))
        #print(Y_predicted)
        print(Y)
        imageio.imwrite('Y_predicted.jpg', Y_predicted)
if __name__ == '__main__':
    i_filter_size = 3 #size of filter
    i_kernel_size = 2 #size of pooling filter
    i_filter = 32 #num of conv filters
    str_optimizer = 'adam'
    str_loss_function = 'mean_square'
    str_activation_function = 'LeakyReLU' 
    f_learning_rate = 0.1
    f_batch_size = 1
    i_epochs = 1
    net = NeuralNetBuild(str_optimizer, f_learning_rate, i_filter_size, i_kernel_size,i_filter,str_activation_function, str_loss_function, i_epochs,f_batch_size)
    run_model(net)
    #import mnistcnn