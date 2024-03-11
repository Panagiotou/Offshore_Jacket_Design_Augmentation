# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:43:11 2020

@author: Utilisateur
"""

from autograd import grad
from .lik_functions import ord_loglik_j, binom_loglik_j, cont_loglik_j, categ_loglik_j

###########################################################################
# Binary/count variables gradient
###########################################################################

def bin_grad_j(lambda_bin_j, y_bin_j, zM, k, ps_y, p_z_ys, nj_bin_j):
    ''' Compute the gradient of the expected log-likelihood for each binomial variable y_j
    
    lambda_bin_j ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin_j (numobs 1darray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_bin (int): The number of possible values/maximum values of the jth binary/count variable
    --------------------------------------------------------------
    returns (float): grad_j(E_{zM, s | y, theta}(y_bin_j | zM, s1 = k1))
    ''' 
    grad_bin_lik = grad(binom_loglik_j)
    return grad_bin_lik(lambda_bin_j, y_bin_j, zM, k, ps_y, p_z_ys, nj_bin_j)

   
###########################################################################
# Ordinal variables gradient
###########################################################################

def ord_grad_j(lambda_ord_j, y_oh, zM, k, ps_y, p_z_ys, nj_ord_j):
    ''' Compute the gradient of the expected log-likelihood for each ordinal variable y_j
    
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The subset containing only the ordinal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_ord_j (int): The number of possible values of the jth ordinal variable
    --------------------------------------------------------------
    returns (float): grad_j(E_{zM, s | y, theta}(y_ord_j | zM, s1 = k1))
    ''' 
    
    grad_ord_lik = grad(ord_loglik_j)
    return grad_ord_lik(lambda_ord_j, y_oh, zM, k, ps_y, p_z_ys, nj_ord_j)

###########################################################################
# Categorical gradient
###########################################################################

def categ_grad_j(lambda_categ_j, y_categ_j, zM, k, ps_y, p_z_ys, nj_categ_j):
    ''' Compute the gradient of the expected log-likelihood for each categorical variable y_j
    
    lambda_categ_j (nj_categ x (r + 1) ndarray): Coefficients of the categorical distributions in the GLLVM layer
    y_categ_j (numobs 1darray): The jth categorical variable in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_categ_j (int): The number of possible values values of the jth categorical variable
    --------------------------------------------------------------
    returns (float): grad_j(E_{zM, s | y, theta}(y_categ_j | zM, s1 = k1))
    ''' 
    
    grad_categ_lik = grad(categ_loglik_j)
    return grad_categ_lik(lambda_categ_j, y_categ_j, zM, k, ps_y, p_z_ys, nj_categ_j)

###########################################################################
# Continuous variables gradient
###########################################################################

def cont_grad_j(lambda_cont_j, y_cont_j, zM, k, ps_y, p_z_ys):
    ''' Compute the gradient of the expected log-likelihood for each continuous variable y_j
    
    lambda_cont_j ( (nj_cont_j + r - 1) 1darray): Coefficients of the continal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The subset containing only the continal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    --------------------------------------------------------------
    returns (float): grad_j(E_{zM, s | y, theta}(y_cont_j | zM, s1 = k1))
    ''' 
    
    grad_cont_lik = grad(cont_loglik_j)
    return grad_cont_lik(lambda_cont_j, y_cont_j, zM, k, ps_y, p_z_ys)