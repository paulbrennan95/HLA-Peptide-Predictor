# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:15:54 2020

@author: Paul Brennan
"""

import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import matplotlib as mpl


# from sklearn.ensemble import RandomForestClassifier


class Peptide:
    def __init__(self, seq):
        self.seq = seq 
        
        
    def change(self):
        return self.seq + ("XXX")
    


# Specify an ordering of amino acids for vectorizing peptide sequence
   
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


dim_tab = pd.read_csv("dimarco_peptides.csv")




def nine_merize(sequence):
    #initialize pep_list
    pep_list = []
    for pep in sequence:
        #select only 9mers
        if len(pep) == 9:            
            pep_list.append(pep)
            
    return pep_list

def label_encode(sequence):
    le = LabelEncoder()
    le.fit_transform(sequence)
    
    
        
def num_encode(sequence):
    ''' input a list list/series of peptides in single letter format to 
    generate a list of numerized peptides'''
    
    #initialize peptide sequence
    pep_list = []
    for pep in sequence:
        
        #initialize amino acid sequence per peptide
        aa_seq = []
        for aa in pep:           
            #change each peptide letter to a number from codes list
            aa_seq.append(int(codes.index(aa)))
            
        #add the numerized peptide to the big list
        pep_list.append(aa_seq)
    
    return np.array(pep_list)
        
def target_vec(array, value):
    return np.array([value]*len(array))

def rand_peptides(length, number):
    rand_list = []
    for pep in range(number):
        rand_list.append(''.join(map(str, (random.sample(codes, k=length))))) 
        # rand_list.append(random.sample(codes, k=length))
    return np.array(rand_list)
    
def Tree_Classifier(
        x_train, x_test, y_train, y_test):
    
        
        tree_list = []
        for value in (2,5):
            clf = DecisionTreeClassifier(max_depth = value, random_state=509)
            clf.fit(x_train, y_train)
            tree_list.append([value, clf.score(x_test, y_test),
                              clf.get_depth()])
        print(np.array(tree_list))
        # PRINT TEXT VERSION OF TREE
        # print(np.array(tree_list))
        # text_representation = tree.export_text(clf)
        # print(text_representation)
        
        #PRINT PNG VERSION OF TREE        
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, 
                   feature_names= ["peptide 1", "peptide 2", "peptide 3",
                                   'peptide 4','peptide 5', 'peptide 6', 
                                   'peptide 7', 'peptide 8', 'peptide 9'],  
                   class_names= ['Non-Binder', 'Binder'],
                   filled=True)
        mpl.rcParams.update({'font.size': 14})
        fig.savefig("decision_tree.png")
        
def Random_Forest_Classifier(
        x_train, x_test, y_train, y_test):
    
        
        tree_list = []
        for value in (2,5,10,50,100):
            clf = RandomForestClassifier(max_depth = value, random_state=509)
            clf.fit(x_train, y_train)
            tree_list.append([value, clf.score(x_test, y_test)])
        
        print(np.array(tree_list))
        
        features = ["p1", "p2", "p3", 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
        importances = clf.feature_importances_
        indices = np.argsort(importances)
        
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

def ANN(x_train, x_test, y_train, y_test, layers = 2, epochs = 20):
    
    #initialize ANN model
    model = Sequential()
    
    # make hidden layers
    for layers in range(layers):
        model.add(Dense(100, activation = 'relu'))
    
    # make output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile, defining loss function and optimization function
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    
    # Fit the training data to the model
    model.fit(x_train, y_train, epochs = epochs)
    
    #Evaluate model with test data
    print("Model Evaluation:   ")
    model.evaluate(x_test, y_test)
    

'''organize train and test data'''


# GENERATE NP ARRAY OF POSITIVE DATA
pos_data = np.array(num_encode(nine_merize(dim_tab['Sequence'])))

# GENERATE NP ARRAY OF RANDOM NEGATIVE DATA
rand_data = num_encode(rand_peptides(9, len(pos_data)))

# GENERATE TARGET VECTORS WITH 1 FOR BINDERS, 0 FOR NON BINDERS 
pos_target = target_vec(pos_data, 1)
rand_target = target_vec(rand_data, 0)


# CONCATENATE POSITIVE AND NEGATIVE ARRAYS 
     #  AND TARGET VECTORS FOR TRAINING AND TESTING
     
combined_data = np.concatenate((pos_data, rand_data))
combined_target = np.concatenate((pos_target, rand_target))

# MAKE TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(
    combined_data, combined_target, train_size=0.8, random_state=509)


'''Run the models!!'''

# ANN(x_train, x_test, y_train, y_test, 5, 20)
Tree_Classifier(x_train, x_test, y_train, y_test)
# Random_Forest_Classifier(x_train, x_test, y_train, y_test)

