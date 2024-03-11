# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:33:27 2020

@author: Utilisateur
"""

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from scipy.special import binom
from sklearn.preprocessing import OneHotEncoder
from .numeric_stability import log_1plusexp, expit, softmax_


def log_py_zM_bin_j(lambda_bin_j, y_bin_j, zM, k, nj_bin_j): 
    ''' Compute log p(y_j | zM, s1 = k1) of the jth binary/count variable
    
    lambda_bin_j ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin_j (numobs 1darray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_bin_j (int): The number of possible values/maximum values of the jth binary/count variable
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''
    M = zM.shape[0]
    r = zM.shape[1]
    numobs = len(y_bin_j)
    
    yg = np.repeat(y_bin_j[np.newaxis], axis = 0, repeats = M)
    yg = yg.astype(float)

    nj_bin_j = float(nj_bin_j)

    coeff_binom = binom(nj_bin_j, yg).reshape(M, 1, numobs, order = 'C')
    
    eta = np.transpose(zM, (0, 2, 1)) @ lambda_bin_j[1:].reshape(1, r, 1, order = 'C') 
    eta = eta + lambda_bin_j[0].reshape(1, 1, 1) # Add the constant
    
    den = nj_bin_j * log_1plusexp(eta)

    num = eta @ y_bin_j[np.newaxis, np.newaxis].astype(float)  
    log_p_y_z = num - den + np.log(coeff_binom)
    
    return np.transpose(log_p_y_z, (0, 2, 1)).astype(float)

def log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin):
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the binomial data with a for loop
    
    lambda_bin (nb_bin x (r + 1) ndarray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_bin (nb_bin x 1d-array): The number of possible values/maximum values of binary/count variables respectively
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1)
    '''
    log_py_zM = 0
    nb_bin = len(nj_bin)
    for j in range(nb_bin):
        log_py_zM += log_py_zM_bin_j(lambda_bin[j], y_bin[:,j], zM, k, nj_bin[j])
        
    return log_py_zM

def binom_loglik_j(lambda_bin_j, y_bin_j, zM, k, ps_y, p_z_ys, nj_bin_j):
    ''' Compute the expected log-likelihood for each binomial variable y_j
    
    lambda_bin_j ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin_j (numobs 1darray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_bin_j (int): The number of possible values/maximum values of the jth binary/count variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_bin_j | zM, s1 = k1)
    ''' 
    log_pyzM_j = log_py_zM_bin_j(lambda_bin_j, y_bin_j, zM, k, nj_bin_j)
    return -np.sum(ps_y * np.sum(p_z_ys * log_pyzM_j, axis = 0))


######################################################################
# Ordinal likelihood functions
######################################################################

def log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord_j): 
    ''' Compute log p(y_j | zM, s1 = k1) of each ordinal variable 
    
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The jth ordinal variable in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord_j (int): The number of possible values values of the jth ordinal variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth ordinal variable
    '''    
    r = zM.shape[1]
    M = zM.shape[0]
    epsilon = 1E-10 # Numeric stability
    lambda0 = lambda_ord_j[:(nj_ord_j - 1)]
    Lambda = lambda_ord_j[-r:]
 
    broad_lambda0 = lambda0.reshape((nj_ord_j - 1, 1, 1, 1))
    eta = broad_lambda0 - (np.transpose(zM, (0, 2, 1)) @ Lambda.reshape((1, r, 1), order = 'C'))[np.newaxis]
    
    gamma = expit(eta)
    
    gamma_prev = np.concatenate([np.zeros((1,M, k, 1)), gamma])
    gamma_next = np.concatenate([gamma, np.ones((1,M, k, 1))])
    pi = gamma_next - gamma_prev
    
    pi = np.where(pi <= 0, epsilon, pi)
    pi = np.where(pi >= 1, 1 - epsilon, pi)
    
    yg = np.expand_dims(y_oh_j.T, 1)[..., np.newaxis, np.newaxis] 
    
    log_p_y_z = yg * np.log(np.expand_dims(pi, axis=2)) 
   
    return log_p_y_z.sum((0))

def log_py_zM_ord(lambda_ord, y_ord, zM, k, nj_ord): 
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the ordinal data with a for loop
    
    lambda_ord ( nb_ord x (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_ord (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord (nb_ord x 1d-array): The number of possible values values of ordinal variables
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1) for ordinal variables
    '''
    
    nb_ord = y_ord.shape[1]

    log_pyzM = 0
    for j in range(nb_ord):
        #enc = OneHotEncoder(categories = [list(range(nj_ord[j]))])
        enc = OneHotEncoder(categories = [np.arange(nj_ord[j]).astype(float)])

        #enc = OneHotEncoder(categories = 'auto')
        #print('j=', j)
        #print([np.arange(nj_ord[j]).astype(float)])
        #print(set(y_ord[:,j][..., n_axis]))
        y_oh_j = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()
        log_pyzM += log_py_zM_ord_j(lambda_ord[j], y_oh_j, zM, k, nj_ord[j])
        
    return log_pyzM
        

def ord_loglik_j(lambda_ord_j, y_oh_j, zM, k, ps_y, p_z_ys, nj_ord_j):
    ''' Compute the expected log-likelihood for each ordinal variable y_j
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The subset containing only the ordinal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_ord_j (int): The number of possible values of the jth ordinal variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_ord_j | zM, s1 = k1)
    ''' 
    #print(y_oh_j.shape)
    #print(p_z_ys.shape)

    log_pyzM_j = log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord_j)
    #print(log_pyzM_j.shape)
    #print('------------------------------')
    return -np.sum(ps_y * np.sum(np.expand_dims(p_z_ys, axis = 3) * log_pyzM_j, (0,3)))


######################################################################
# Categorical likelihood functions
######################################################################

# nj_categ_j is useless and could be removed in future versions
def log_py_zM_categ_j(lambda_categ_j, y_categ_j, zM, k, nj_categ_j):
    ''' Compute log p(y_j | zM, s1 = k1) of each categorical variable 
    
    lambda_categ_j (nj_categ x (r + 1) ndarray): Coefficients of the categorical distributions in the GLLVM layer
    y_categ_j (numobs 1darray): The jth categorical variable in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_categ_j (int): The number of possible values values of the jth categorical variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth categorical variable
    '''  
    epsilon = 1E-10

    r = zM.shape[1]
    nj = y_categ_j.shape[1]
        
    zM_broad = np.expand_dims(np.expand_dims(np.transpose(zM, (0, 2, 1)), 2), 3)
    lambda_categ_j_ = lambda_categ_j.reshape(nj, r + 1, order = 'C')

    eta = zM_broad @ lambda_categ_j_[:, 1:][n_axis, n_axis, ..., n_axis] # Check que l'on fait r et pas k ?
    eta = eta + lambda_categ_j_[:,0].reshape(1, 1, nj_categ_j, 1, 1, order = 'C')  # Add the constant
    
    pi = softmax_(eta.astype(float), axis = 2)
    
    # Numeric stability
    pi = np.where(pi <= 0, epsilon, pi)
    pi = np.where(pi >= 1, 1 - epsilon, pi)

    yg = np.expand_dims(np.expand_dims(y_categ_j, 1), 1)[..., np.newaxis, np.newaxis] 
    log_p_y_z = yg * np.log(pi[n_axis]) 
    
    # Reshaping output
    log_p_y_z = log_p_y_z.sum((3)) # Suming over the modalities nj
    log_p_y_z = log_p_y_z[:,:,:,0,0] # Deleting useless axes
        
    return np.transpose(log_p_y_z,(1,0, 2))

def log_py_zM_categ(lambda_categ, y_categ, zM, k, nj_categ):
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the categorical data with a for loop
    
    lambda_categ (list of nj_categ x (r + 1) ndarrays): Coefficients of the categorical distributions in the GLLVM layer
    y_categ (numobs x nb_categ ndarray): The subset containing only the categ variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_categ (nb_categ x 1d-array): The number of possible values categorical variables respectively
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1)
    '''
    log_py_zM = 0
    nb_categ = len(nj_categ)
    #enc = OneHotEncoder(categories='auto')
    
    for j in range(nb_categ):
        classes = [str(nj) for nj in range(nj_categ[j])]
        
        enc = OneHotEncoder(categories = [classes])
        y_categ_j = enc.fit_transform(y_categ[:,j][..., n_axis]).toarray()
        log_py_zM += log_py_zM_categ_j(lambda_categ[j], y_categ_j, zM, k, nj_categ[j])
        
    return log_py_zM



def categ_loglik_j(lambda_categ_j, y_categ_j, zM, k, ps_y, p_z_ys, nj_categ_j):
    ''' Compute the expected log-likelihood for each categ variable y_j
    lambda_categ_j (nj_categ x (r + 1) ndarray): Coefficients of the categorical distributions in the GLLVM layer
    y_categ_j (numobs 1darray): The jth categorical variable in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_categ_j (int): The number of possible values values of the jth categorical variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_categ_j | zM, s1 = k1)
    ''' 
    r = zM.shape[1]
    nj = y_categ_j.shape[1]
    
    # Ensure the good shape of the input
    lambda_categ_j_ = lambda_categ_j.reshape(nj, r + 1, order = 'C')
    
    log_pyzM_j = log_py_zM_categ_j(lambda_categ_j_, y_categ_j, zM, k, nj_categ_j)
    return -np.sum(ps_y * np.sum(np.expand_dims(p_z_ys, axis = 3) * log_pyzM_j[..., n_axis], (0,3)))



######################################################################
# Continuous likelihood functions
######################################################################

def log_py_zM_cont(lambda_cont, y_cont, zM, k):
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the continuous data with a for loop
    
    lambda_cont (nb_cont x (r + 1) ndarray): Coefficients of the continuous distributions in the GLLVM layer
    y_cont (numobs x nb_cont ndarray): The subset containing only the continuous variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    --------------------------------------------------------------
    returns (ndarray): sum_j p(y_j | zM, s1 = k1)
    '''
    log_pyzM = 0
    nb_cont = y_cont.shape[1]
    
    for j in range(nb_cont):
        log_pyzM += log_py_zM_cont_j(lambda_cont[j], y_cont[:,j], zM, k)
        
    return log_pyzM

def log_py_zM_cont_j(lambda_cont_j, y_cont_j, zM, k):
    ''' Compute log p(y_j | zM, s1 = k1) of the jth continuous variable
    
    lambda_cont_j ( (r + 1) 1darray): Coefficients of the continuous distributions in the GLLVM layer
    y_cont_j (numobs 1darray): The subset containing only the continuous variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''
    r = zM.shape[1]
    M = zM.shape[0]

    yg = np.repeat(y_cont_j[np.newaxis], axis = 0, repeats = M)
    yg = np.expand_dims(yg, 1)
    
    eta = np.transpose(zM, (0, 2, 1)) @ lambda_cont_j[1:].reshape(1, r, 1, order = 'C')
    eta = eta + lambda_cont_j[0].reshape(1, 1, 1) # Add the constant
    
    return t(- 0.5 * (np.log(2 * np.pi) + (yg - eta) ** 2), (0, 2, 1))

def cont_loglik_j(lambda_cont_j, y_cont_j, zM, k, ps_y, p_z_ys):
    ''' Compute the expected log-likelihood for each continuous variable y_j
    
    lambda_cont_j ( (r + 1) 1darray): Coefficients of the continuous distributions in the GLLVM layer
    y_cont_j (numobs 1darray): The subset containing only the continuous variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_bin_j | zM, s1 = k1)
    ''' 
    log_pyzM_j = log_py_zM_cont_j(lambda_cont_j, y_cont_j, zM, k)
    return -np.sum(ps_y * np.sum(p_z_ys * log_pyzM_j, axis = 0))


