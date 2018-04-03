# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:08:09 2018

@author: NILESH
"""
import numpy as nu
import utility_module_NN
def find_dot_product(a,b):
    m,n = nu.shape(a)
    vector = nu.zeros((m,n),dtype=nu.float64)
    for i in range(0,m):
        vector[i][0] = a[i][0] * b[i][0]
    return vector
def backpropogation_algorithm(X , Y, weights_dictionary, config_list_per_layer, config_list_size,lamda):
    try:
        gradient_calcuated = {}
        m = len(X)
        for i in range(0,config_list_size - 1):
            key = "gradient_" + str(i)
            gradient_calcuated[key] = nu.zeros((config_list_per_layer[i] + 1 ,config_list_per_layer[i+1]),dtype = nu.float64)
            
        for i in range (0,m):
            hypothesis_output = {}
            input_instance = X[i][:]
            hypothesis_output = utility_module_NN.feed_forward_NN(weights_dictionary,input_instance,config_list_per_layer,config_list_size)
            last_layer_key = 'output_' + str(config_list_size - 2)
            final_layer_hypothesis = hypothesis_output[last_layer_key]
            delta_last_layer = nu.subtract(final_layer_hypothesis,Y[i][:])
            delta = [delta_last_layer]
            for j in range(config_list_size - 1, 0 , -1): #This is for hidder layers
                key_for_hidden_layer = 'gradient_' + str(j-1)
                if(j == 1):
                    output = [X[i]]
                else:
                    key_for_output = 'output_' + str(j-2)
                    output =  [hypothesis_output[key_for_output]]               
                intermidiate_derivative = nu.matmul(nu.transpose(output) , delta)
                gradient_calcuated[key_for_hidden_layer] = gradient_calcuated[key_for_hidden_layer] + intermidiate_derivative
                sigmoid_gradient = nu.transpose(utility_module_NN.gradient_sigmoid_function(output))
                key_weights = 'weights_' + str(j-1)
                weights_for_hidden_layer = weights_dictionary[key_weights]
                delta_intermediate = nu.matmul(weights_for_hidden_layer,nu.transpose(delta))
                #new_delta = [a * b for a,b in zip(delta_intermediate[:],sigmoid_gradient[:])]
                #new_delta_1 = nu.dot(delta_intermediate[:],sigmoid_gradient[:])
                new_delta = find_dot_product(delta_intermediate[:],sigmoid_gradient[:])
                delta = [nu.transpose(new_delta)[0][1::]]
                
         
        for i in range(0,config_list_size - 1):
            key = "gradient_" + str(i)
            key_weight = 'weights_' + str(i)
            weights = weights_dictionary[key_weight]
            new_ones_mat = nu.ones((config_list_per_layer[i],config_list_per_layer[i+1]),dtype=nu.float64)
            zeros_vector = nu.zeros((1,config_list_per_layer[i+1]),dtype=nu.float64)
            matrix = nu.append(zeros_vector,new_ones_mat,axis=0) * lamda
            regularized_weights = weights * matrix
            temp = (gradient_calcuated[key] * (1/m)) + ((1/m)* regularized_weights)
            gradient_calcuated[key] = temp
        return gradient_calcuated     
    except Exception:
        print("Issue in finding gradient using backpropogation algorithm")
    