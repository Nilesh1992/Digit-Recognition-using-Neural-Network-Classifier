# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:48:23 2018

@author: NILESH
"""
import scipy.io as spio
import os
import numpy
import utility_module_NN
file_path = os.getcwd() + "\ex4data1.mat"
mat = spio.loadmat(file_path, squeeze_me=True) #This is to load the matrix
X = mat['X']
Y = mat['y']
length = len(X)
new_feature = numpy.ones((length,1),dtype=numpy.float64)
#axis = 0 means add in as row
#asis = 1 means add in as column
X = numpy.append(new_feature,X,axis = 1)
temp_X = X
temp_Y = Y
output = utility_module_NN.map_the_output(Y)
iterations = 10000
learning_rate = 0.03
lamda = 1
#Our NN will contain neurons 25 hidden layers, 10 neurons in output layer
#configuration is [400,25,10]
config_list_per_layer = [400,25,10] 
config_list_size = len(config_list_per_layer)
weights_dictionary_1 = utility_module_NN.initalise_weights_to_break_symmetry(config_list_per_layer,config_list_size)
J_history,weights_dictionary = utility_module_NN.gradient_descent_for_function_minimization(iterations, learning_rate, X, output, weights_dictionary_1, lamda, config_list_per_layer, config_list_size)
#weights_dictionary_neumerical = utility_module_NN.numerical_gradient_checking(X, output, weights_dictionary_1,lamda, config_list_per_layer, config_list_size)
accuracy = utility_module_NN.predict_class_with_accuracy(temp_X,temp_Y,weights_dictionary,config_list_per_layer,config_list_size)