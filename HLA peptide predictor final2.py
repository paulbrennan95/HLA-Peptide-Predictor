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



from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt


# these amino acids codes will serve to 
#   encode letters into numbers (from indices)
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        

class PeptideFile:
    ''' Import csv file with list of peptide sequences 
        in single column and no header'''
    
    def __init__(self, filename, filetype = 'csv'):
        self.filename = filename
        self.filetype = filetype
        
        if self.filetype == 'csv':
            pd_data = pd.read_csv(self.filename, header = None)
            self.seqs = np.array(pd_data)
            # print(self.seqs)
            #return np.array(pd_data)
        
        else:
            print("Error: Please enter a CSV file.")
     
            pass
        
        
        self.num_encoded = num_encode(self.seqs)
        self.label_encoded = label_encode(self.seqs)
        self.nine_mers = nine_merize(self.seqs)
    
    
def nine_merize(array):
    ''' make a single column array of peptides of various length into an
        array of peptides with length 9
        
        Returns:  numpy array of nonamer peptides'''
        
    #initialize pep_list
    pep_list = []
    for row in array:
        for pep in row:
            #print(pep)
            row_list= []
            #select only 9mers                  
            if len(pep) == 9: 
                
                row_list.append(pep)
                pep_list.append(row_list)
        
    
    return np.array(pep_list)
           
    
    
def add_target_vec(length, value):
    '''creates vector of single value (say, 0 or 1) '''
    return TargetVector(np.array([value]*length))



class TargetVector:
    '''class to manipulate/modify target vector'''
    def __init__(self, target_vector):
        self.target = target_vector
        self.length = len(target_vector)
        

   
        
def num_encode(array):
    ''' input a single column array of peptides in single letter format to 
    generate a list of numerized peptides
    
    returns:  single column array of number-encoded peptides '''
    
    #initialize peptide sequence
    pep_list = []
    for row in array:
        
        for pep in row:
            
            #initialize amino acid sequence per peptide
            aa_seq = []
            for aa in pep:
                
                #change each peptide letter to a number from codes list
                aa_seq.append(int(codes.index(aa)))
                
            
                
            #add the numerized peptide to the big list
            pep_list.append(aa_seq)
            
    return np.array(pep_list)

def label_encode(array):   
    ''' label encodes (alternative to num_encode) array
        prior to Random Forest Classification'''    
    return LabelEncoder().fit_transform(array)

def rand_peptides(length, number):
    '''generates given number of random peptides of given length
    Args: 
        length: length of desired peptides
        number: number of desired peptides
        
    Returns: single column array of random peptides '''
    
    rand_list = []
    for pep in range(number):
        row_list = []
        #makes list of random letters of said length, and makes string
        row_list.append(''.join(map(str, (random.sample(codes, k=length))))) 
        rand_list.append(row_list)
        
    return np.array(rand_list)
    
     
def Random_Forest_Classifier(
        x_train, x_test, y_train, y_test, pred_seq):
    
    '''uses sklearn RandomForestClassifier package to fit and train and test
        data processed by the train_test_split() function
        
        Args:
            x_train:  input training set
            x_test:  testing set to determine accuracy
            y_train: target vector to train x_train
            y_test: target vector to test x_test
            pred_seq: array of peptides to predict HLA-C*16:01 binding
            
        Returns:
            Score of training set
            Array of probability predictions for prediction sequences
            Plot of Feature Importances of Peptide Positions'''
        
        
    #instantiate RFC class, max_depth of 50 was optimal
    rf = RandomForestClassifier(max_depth = 50, 
                                 random_state=509)
    #fit model
    rf.fit(x_train, y_train)
    
    #round scoring digits to hundreths place
    np.set_printoptions(precision=3)
    
    #score testing data to determine efficacy/accuracy of model
    score = rf.score(x_test, y_test)

    print('Training Set Score:  ', "{:.2f}".format(score))
    
    #Generate graph of feature importances
    #   (which peptide positions dictate the binding motif)
    features = ["p1", "p2", "p3", 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    
    return rf.predict_proba(pred_seq)
        


def main(file,
         algorithm,
         prediction_file):
    '''main function organizing data from input files. It isolates 9-mers, 
    encodes them into numbers, and splits it into train and test sets. It 
    generates a list of random peptide sequences to serve as negative training
    set. It produces target vectors to train and test accuracy of model.
    
    Args:
        file: input csv file containing peptide sequences in first column with 
                header
        algorithm: RandomForest is the only algorithm for now, but future modules
                    will contain other options
        prediction_file:  csv file of peptides to predict binding probabilities
                based on model generating by input training data file
    
    Returns:
        output_file.txt saved in program directory with 
            prediction peptides and their binding probabilities'''
    
        
    # import file     
    file = PeptideFile(file)
       
    # get sequence data from file
    data = num_encode(nine_merize(file.seqs))  
      
    # GENERATE NP ARRAY OF POSITIVE DATA
    pos_data = data
  
    # GENERATE NP ARRAY OF RANDOM NEGATIVE DATA
    rand_data = num_encode(rand_peptides(9, len(pos_data)))
  
    # GENERATE TARGET VECTORS WITH 1 FOR BINDERS, 0 FOR NON BINDERS 
    pos_target = add_target_vec(len(pos_data), 1).target.ravel()
    rand_target = add_target_vec(len(rand_data), 0).target.ravel()
   
    # CONCATENATE POS+NEG ARRAYS and TARGET VECTORS FOR TRAINING AND TESTING
    combined_data = np.concatenate((pos_data, rand_data))
    combined_target = np.concatenate((pos_target, rand_target))
    
    # MAKE TRAIN AND TEST SETS
    x_train, x_test, y_train, y_test = train_test_split(
        combined_data, 
        combined_target, 
        train_size=0.8, 
        random_state=509)
      
    # SET UP PREDICTION FILE
    if prediction_file:
               
        # import file     
        pred_file = PeptideFile(prediction_file)
        
        # prediction 9mers only 
        pred_ninemers = nine_merize(pred_file.seqs)
        
        # get sequence data from file
        pred_data = num_encode(pred_ninemers)
         
        
    
    # print training module info
    print("\nTRAINING MODULE:  {} in {} Analysis ".format(file.filename,
                                                  algorithm))
    
    # run algorithm 
    if algorithm == 'RandomForest':
        outcome = Random_Forest_Classifier(x_train, x_test, y_train, y_test, pred_data)
        
        
    else:
        print("Select algorithm: For now, only 'RandomForest' is available.")
        
    
    #make output file
    if prediction_file and algorithm == 'RandomForest':
        
        #add prediction probabilities to letter-coded peptides
        output = np.insert(pred_ninemers, 1, outcome[:,1], axis=1)
        
        #sort on probabilities
        sort_output = output[output[:,1].argsort()]
        
        #flip the order to show high-binders first
        final_output = np.flip(sort_output, axis=0)
    
        
        #write the resulting output array to a file
        with open('output_file.txt', 'w') as new_file:
            new_file.write("Prediction Results for {} \n\n".format(prediction_file))
            for row in final_output:
                new_file.write(str(row[0]) + '  ' + str(row[1]) + '\n')
                
         
        print('Prediction Results saved in output_file.txt\n'+
              'Check Plots for Feature Importances Graph of Training Set'
              )
    
if __name__ == "__main__":
    # main("sarkizova_peptides.csv", 'RandomForest')
    # main('C1601_peptides_all.csv', 'RandomForest')

    main("C1601_peptides_all.csv", 
         'RandomForest',
         "Random_Peptide_file2.csv" )    


