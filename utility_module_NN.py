# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:52:02 2018

@author: NILESH
"""
import numpy as nu
import backpropogation
import copy
#This inialize the weights in range [-epsilon, +epsilon] for specific input and output layer in NN
def initalise_weights_to_break_symmetry(config_list_per_layer,config_list_len):
    try:
        input_layer_nurons = config_list_per_layer[0]
        weights_dictionary = {}
        output_layer_nurons = config_list_per_layer[config_list_len-1]
        epsilon = nu.sqrt(6)/nu.sqrt(input_layer_nurons + output_layer_nurons) 
        for i in range(0,config_list_len - 1):
            key = 'weights_' + str(i)
            L_in = int(config_list_per_layer[i])
            L_out = int(config_list_per_layer[i + 1])
            weights_dictionary[key] = (nu.random.rand(L_in + 1, L_out) * 2 * epsilon) - epsilon 
        return weights_dictionary
    except Exception:
        print("Some issue while weight initalisation in function initalise_weights_to_break_symmetry")

#This method return gradient of learned features 
# h(z) = 1/(1 + e^(-z)) is sigmoid finction 
# derivative with respect to z d/dz (h(z)) = h(z)(1 - h(z))        
def gradient_sigmoid_function(learned_input):
    try:
        learned_nigation = nu.subtract(1,learned_input)
        return [i * j for i,j in zip(learned_input,learned_nigation)]
    except Exception:  
        print("Some issue while taking derivative, in gradient_sigmoid_function")


#Input: intermediate output
#Sigmod function: 1/1+e^(-(intermediate_output)) 
def get_sigmoid_hypothesis(hypothesis_output_intermidiate):
    try:
        return (1/(1+nu.exp(-hypothesis_output_intermidiate)))
    except Exception:
        print("Issue in getting hypothesis\n 1)Please check the dimensions ")
            

def feed_forward_NN(weights_dictionary,input_instance,config_list_per_layer,config_list_size):
    try:
        hypothesis_dictionary = {}
        learned_feature = input_instance
        for i in range(0,config_list_size-1):
            key = 'weights_' + str(i)
            key_for_hypothesis_per_layer = 'output_' + str(i)
            weights_for_layer = weights_dictionary[key]
            intermediate_learned_feature = nu.matmul(nu.transpose(learned_feature),weights_for_layer)
            learned_feature = get_sigmoid_hypothesis(intermediate_learned_feature)
            if(i != config_list_size - 2):
                learned_feature = nu.append([1], learned_feature, axis=0) #adding bias term
            hypothesis_dictionary[key_for_hypothesis_per_layer] = learned_feature
        return hypothesis_dictionary
    except Exception:
        print("Issue in getting output for the last layer in feed_forward_NN")

def get_modified_output_to_avoid_infinity(hypothesis):
    try:
        length = len(hypothesis)
        for i in range(0,length-1):
            if(hypothesis[i] == 1):
                hypothesis[i] = 0.99999999999
            if(hypothesis[i] == 0):
                hypothesis[i] = 0.00000000001
        return hypothesis    
    except Exception:
        print("Issue while modifying the hypothesis")

def get_cost_for_current_instance(weights_dictionary, input_instance, output_instance, config_list_per_layer,config_list_size):
    try:
        hypothesis_dictionary = feed_forward_NN(weights_dictionary,input_instance,config_list_per_layer,config_list_size)
        key_for_last_layer_hypothesis = 'output_' + str(config_list_size - 2)
        hypothesis_last_layer = get_modified_output_to_avoid_infinity(hypothesis_dictionary[key_for_last_layer_hypothesis])
        #This previous step is important since the log(1) and log(0) outputs infinity, which we don't want
        left_term = nu.sum([i * j for i,j in zip(output_instance,nu.log(hypothesis_last_layer))])
        right_term = nu.sum([(1-i) * j for i,j in zip(output_instance,nu.log(1-hypothesis_last_layer))])
        return (left_term + right_term)
    except Exception:
        print("Issue in getting cost for sepcific instance")

def get_cost_for_regularization(weights_dictionary, config_list_size):
    try:
        total_regularization_cost = 0
        for i in range(0,config_list_size-1):
            key = 'weights_' + str(i)
            weights = weights_dictionary[key]
            total_regularization_cost = total_regularization_cost + nu.sum(nu.square(weights[1::][:])) 
        return total_regularization_cost   
    except Exception: 
        print("Issue in getting cost for regularized terms")

def loss_function_with_regulariztion(X, Y, weights_dictionary, lamda, config_list_per_layer, config_list_size):
    try:
       total_number_of_traning_example = len(Y)
       current_cost = 0
       regularized_term = (lamda/(2*total_number_of_traning_example))* get_cost_for_regularization(weights_dictionary, config_list_size)
       for i in range(0,total_number_of_traning_example):   
           output_for_current_instance = get_cost_for_current_instance(weights_dictionary, X[i][:] , Y[i][:], config_list_per_layer,config_list_size)               
           current_cost = current_cost + output_for_current_instance
       current_cost = (-(1/total_number_of_traning_example)*current_cost) + regularized_term
       print(current_cost)
       return current_cost
    except Exception:
        print("Issue in getting output for the parameters in loss fuction")


def gradient_descent_for_function_minimization(iteration, learning_rate, X, Y, weights_dictionary, lamda, config_list_per_layer, config_list_size):
    J_history = nu.zeros((iteration,1),dtype=nu.float64)
    for i in range(0,iteration):
        print('Iteration:',i)
        J_history[i,0] = loss_function_with_regulariztion(X, Y, weights_dictionary, lamda, config_list_per_layer, config_list_size)
        gradient_dictionary = backpropogation.backpropogation_algorithm(X , Y, weights_dictionary, config_list_per_layer, config_list_size,lamda)
        for i in range(0,config_list_size-1):
            key = 'gradient_' + str(i)
            weight_key = 'weights_' + str(i)
            gradients = gradient_dictionary[key]
            weights = weights_dictionary[weight_key]
            temp = weights - (learning_rate*(gradients))
            weights_dictionary[weight_key] = temp
    return J_history,weights_dictionary   
               
def map_the_output(Y):
    try:
        total_number_of_labels = len(Y)
        max_label = nu.max(Y)
        out_mat = nu.zeros((total_number_of_labels, max_label),dtype=nu.float64)
        for i in range(0,total_number_of_labels):
            index = Y[i]
            if(index == 10):
                index = 0
            out_mat[i][index] = 1
        return out_mat
    except Exception:
        print("issue while mapping the output, in function map_the_output")
#Just for test
#output = initalise_weights_to_break_symmetry(3,[400,25,10],3)
#out = get_sigmoid_hypothesis([[1,2,3]],[[-2,-3,1],[-2,-3,1]])
#output_1 = gradient_sigmoid_function(out)

#This is the way we can check if the backpropogation algorithm implementaion is correct, to be precise we can use for any ML algo 
def numerical_gradient_checking(X, Y, weight_dictonary,lamda, config_list_per_layer, config_list_size):
    try:
        gradient_weights = copy.deepcopy(weight_dictonary)
        positive_weight_dictonary = {}
        negative_weight_dictonary = {}
        epislon = float(0.0001)
        for i in range(0,config_list_size-1):   
            weight_key = 'weights_' + str(i-1)
            weights = weight_dictonary[weight_key]
            row,col = nu.shape(weights)
            for k in range(0,row):
                for l in range(0,col):
                    positive_weight_dictonary = {}
                    negative_weight_dictonary = {}
                    positive_weight_dictonary = copy.deepcopy(weight_dictonary)
                    negative_weight_dictonary = copy.deepcopy(weight_dictonary)
                    positive_weight_dictonary[weight_key][k][l] = positive_weight_dictonary[weight_key][k][l] + epislon 
                    negative_weight_dictonary[weight_key][k][l] = negative_weight_dictonary[weight_key][k][l] - epislon
                    pos_cost = loss_function_with_regulariztion(X, Y, positive_weight_dictonary, lamda, config_list_per_layer, config_list_size)
                    neg_cost = loss_function_with_regulariztion(X, Y, negative_weight_dictonary, lamda, config_list_per_layer, config_list_size)
                    gradient_weights[weight_key][k][l] = (pos_cost - neg_cost)/ (2*epislon)
                    print(gradient_weights[weight_key][k][l])
        return gradient_weights            
    except Exception:
        print("Issue while calculating gradient using neumerical gradient checking...")
        

def predict_class_with_accuracy(X,Y,weights_dictionary,config_list_per_layer,config_list_size):
    try:
        m = len(X)
        correct_class = int(0)
        for i in range(0,m):
            predicted_hypothesis = copy.deepcopy(feed_forward_NN(weights_dictionary,X[i][:],config_list_per_layer,config_list_size))
            output_key = 'output_1' 
            predicted = predicted_hypothesis[output_key]
            predected_class_index = nu.argmax(predicted)
            if(predected_class_index == Y[i]%10):
                correct_class = correct_class + 1       
        return correct_class     
    except Exception:
        print("Issue While calaulating traning accuracy")
     
        