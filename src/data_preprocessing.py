# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:15:20 2020

@author: rfuchs
"""

from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 

import pandas as pd

import autograd.numpy as np
from autograd.numpy.random import uniform
from autograd.numpy import newaxis as n_axis


def gen_categ_as_bin_dataset(y, var_distrib):
    ''' Convert the categorical variables in the dataset to binary variables
    
    y (numobs x p ndarray): The observations containing categorical variables
    var_distrib (p 1darray): An array containing the types of the variables in y 
    ----------------------------------------------------------------------------
    returns ((numobs, p_new) ndarray): The new dataset where categorical variables 
    have been converted to binary variables
    '''
    new_y = deepcopy(y)
    new_y = new_y.reset_index(drop = True)
    new_var_distrib = deepcopy(var_distrib[var_distrib != 'categorical'])

    categ_idx = np.where(var_distrib == 'categorical')[0]
    oh = OneHotEncoder(drop = 'first')
        
    for idx in categ_idx:
        name = y.iloc[:, idx].name
        categ_var = pd.DataFrame(oh.fit_transform(pd.DataFrame(y.iloc[:, idx])).toarray())
        nj_var = len(categ_var.columns)
        categ_var.columns = [str(name) + '_' + str(categ_var.columns[i]) for i in range(nj_var)]
        
        # Delete old categorical variable & insert new binary variables in the dataframe
        del(new_y[name])
        new_y = new_y.join(categ_var.astype(int))
        new_var_distrib = np.concatenate([new_var_distrib, ['bernoulli'] * nj_var])
        
    return new_y, new_var_distrib

def ordinal_encoding(sequence, ord_labels, codes):
    ''' Perform label encoding, replacing ord_labels with codes
    
    sequence (numobs 1darray): The sequence to encode
    ord_labels (nj_ord_j 1darray): The labels existing in sequences 
    codes (nj_ord_j 1darray): The codes used to replace ord_labels 
    -----------------------------------------------------------------
    returns (numobs 1darray): The encoded sequence
    '''
    new_sequence = deepcopy(sequence.values)
    for i, lab in enumerate(ord_labels):
        new_sequence = np.where(new_sequence == lab, codes[i], new_sequence)

    return new_sequence
    
def compute_nj(y, var_distrib):
    ''' Compute nj for each variable y_j
    
    y (numobs x p ndarray): The original data
    var_distrib (p 1darray): The type of the variables in the data
    -------------------------------------------------------------------
    returns (tuple (p 1d array, nb_bin 1d array, nb_ord 1d array)): The number 
    of categories of all the variables, for count/bin variables only and for 
    ordinal variables only
    '''
    
    nj = []
    nj_bin = []
    nj_ord = []
    nj_categ = []
    
    for i in range(len(y.columns)):
        if np.logical_or(var_distrib[i] == 'bernoulli', var_distrib[i] == 'binomial'): 
            max_nj = int(np.max(y.iloc[:,i], axis = 0))
            nj.append(max_nj)
            nj_bin.append(max_nj)
        elif var_distrib[i] == 'ordinal':
            card_nj = len(np.unique(y.iloc[:,i]))
            nj.append(card_nj)
            nj_ord.append(card_nj)
        elif var_distrib[i] == 'categorical':
            card_nj = len(np.unique(y.iloc[:,i]))
            nj.append(card_nj)
            nj_categ.append(card_nj)            
        elif var_distrib[i] == 'continuous':
            nj.append(np.inf)
        else:
            raise ValueError('Data type', var_distrib[i], 'is illegal')

    nj = np.array(nj)
    nj_bin = np.array(nj_bin)
    nj_ord = np.array(nj_ord)
    nj_categ = np.array(nj_categ)

    return nj, nj_bin, nj_ord, nj_categ

def bin_to_bern(Nj, yj_binom, zM_binom):
    ''' Split the binomial variable into Bernoulli. Them just recopy the corresponding zM.
    It is necessary to fit binary logistic regression
    Example: yj has support in [0,10]: Then if y_ij = 3 generate a vector with 3 ones and 7 zeros 
    (3 success among 10).
    
    Nj (int): The upper bound of the support of yj_binom
    yj_binom (numobs 1darray): The Binomial variable considered
    zM_binom (numobs x r nd-array): The continuous representation of the data
    -----------------------------------------------------------------------------------
    returns (tuple of 2 (numobs x Nj) arrays): The "Bernoullied" Binomial variable
    '''
    
    n_yk = len(yj_binom) # parameter k of the binomial
    
    # Generate Nj Bernoullis from each binomial and get a (numobsxNj, 1) table
    u = uniform(size =(n_yk,Nj))
    p = (yj_binom/Nj)[..., n_axis]
    yk_bern = (u > p).astype(int).flatten('A')#[..., n_axis] 
        
    return yk_bern, np.repeat(zM_binom, Nj, 0)

def data_processing(y, var_distrib, cast_types = False):
    dtypes_dict = {'continuous': float, 'categorical': str, 'ordinal': float,\
                  'bernoulli': int, 'binomial': int}
        
    p = y.shape[1]
    le_dict = {}
    
    df = deepcopy(y)
    #===========================================#
    # Formating the data
    #===========================================#
                            
    # Encode non-continuous variables
    for col_idx, colname in enumerate(df.columns):
        if np.logical_and(var_distrib[col_idx] != 'continuous', var_distrib[col_idx] != 'binomial'): 

            le = LabelEncoder()
            df[colname] = le.fit_transform(df[colname])
            le_dict[colname] = deepcopy(le)
        
    # Feature category (cf)
    if cast_types:
        dtype = {df.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p)}
        df = df.astype(dtype)
    
    return df, le_dict

