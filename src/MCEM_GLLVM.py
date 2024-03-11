# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:51:26 2020

@author: rfuchs
"""

from .lik_functions import ord_loglik_j, log_py_zM_ord, \
            log_py_zM_bin, binom_loglik_j, log_py_zM_cont, cont_loglik_j,\
            log_py_zM_categ, categ_loglik_j
from .lik_gradients import ord_grad_j, bin_grad_j, cont_grad_j, categ_grad_j

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm

from copy import deepcopy
import autograd.numpy as np 
from autograd.numpy import newaxis as n_axis

import warnings
#=============================================================================
# MC Step functions
#=============================================================================

def draw_zl1_ys(z_s, py_zl1, M):
    ''' Draw from p(z1 | y, s) proportional to p(y | z1) * p(z1 | s) for all s 
    z_s (list of nd-arrays): zl | s^l for all s^l and all l.
    py_zl1 (nd-array): p(y | z1_M) 
    M (list of int): The number of MC points on all layers
    ------------------------------------------------------------------------
    returns ((M1, numobs, r1, S1) nd-array): z^{(1)} | y, s
    '''
    epsilon = 1E-16
    
    numobs = py_zl1.shape[1]
    L = len(z_s) - 1
    S = [z_s[l].shape[2] for l in range(L)]
    r = [z_s[l].shape[1] for l in range(L + 1)]

    norm_cste = np.sum(py_zl1, axis = 0, keepdims = True) 
    norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 
    
    py_zl1_norm = py_zl1 / norm_cste
        
    zl1_ys = np.zeros((M[0], numobs, r[0], S[0]))
    for s in range(S[0]):
        qM_cum = py_zl1_norm[:,:, s].T.cumsum(axis=1)
        u = np.random.rand(numobs, 1, M[0])
        
        choices = u < qM_cum[..., np.newaxis]
        idx = choices.argmax(1)
        
        zl1_ys[:,:,:,s] = np.take(z_s[0][:,:, s], idx.T, axis=0)
        
    return zl1_ys

#=============================================================================
# E Step functions
#=============================================================================


def fy_zl1(lambda_bin, y_bin, nj_bin, lambda_ord, y_ord, nj_ord, lambda_categ,\
           y_categ, nj_categ, y_cont, lambda_cont, zl1_s):
    ''' Compute log p(y | z1) = sum_{s= 1}^S[0] p(y, s| z1) as in Cagnone and 
    Viroli (2014)
    lambda_bin (nb_bin x (1 + r1) nd-array): The binomial coefficients
    y_bin (numobs x nb_bin nd-array): The binary/count data
    nj_bin (list of int): The number of modalities for each bin variable
    lambda_ord (list of nb_ord_j x (nj_ord + r1) elements): The ordinal coefficients
    y_ord (numobs x nb_ord nd-array): The ordinal data
    nj_ord (list of int): The number of modalities for each ord variable
    zl1_s ((M1, r1, s1) nd-array): z1 | s 
    ------------------------------------------------------------------------------
    returns ((M1, numobs, S1) nd-array):log p(y | z1_M)
    '''
    
    M0 = zl1_s.shape[0]
    S0 = zl1_s.shape[2] 
    numobs = len(y_bin)
    
    nb_ord = len(nj_ord)
    nb_bin = len(nj_bin)
    nb_categ = len(nj_categ)
    nb_cont = y_cont.shape[1]
     
    log_py_zl1 = np.zeros((M0, numobs, S0), dtype = float) # l1 standing for the first layer
    
    if nb_bin: # First the Count/Binomial variables
        log_py_zl1 += log_py_zM_bin(lambda_bin, y_bin, zl1_s, S0, nj_bin) 
    
    if nb_ord: # Then the ordinal variables 
        log_py_zl1 += log_py_zM_ord(lambda_ord, y_ord, zl1_s, S0, nj_ord)[:,:,:,0] 
        
    if nb_categ:
        log_py_zl1 += log_py_zM_categ(lambda_categ, y_categ, zl1_s, S0, nj_categ) 
        
    if nb_cont:
        log_py_zl1 += log_py_zM_cont(lambda_cont, y_cont, zl1_s, S0)

                    
    py_zl1 = np.exp(log_py_zl1)
    py_zl1 = np.where(py_zl1 == 0, 1E-50, py_zl1)
            
    return py_zl1


def E_step_GLLVM(zl1_s, mu_l1_s, sigma_l1_s, w_s, py_zl1):
    ''' Compute the distributions involved involved in the E step of 
    the GLLVM coefficients estimations
    zl1_s ((M1, r1, s1) nd-array): z1 | s 
    mu_l1_s (nd-array): mu_s for all s in S1 (mu_s starting from the 1st layer)
    sigma_l1_s (nd-array): sigma_s for all s in S1 (sigma_s starting from the 1st layer)
    w_s (list of length s1): The path probabilities for all s in S1
    py_zl1 (nd-array): p(y | z1_M)
    ----------------------------------------------------------------------------
    returns (tuple of len 3): p(z1 |y, s), p(s |y) and p(y)
    '''
    epsilon = 1E-16

    M0 = zl1_s.shape[0]
    S0 = zl1_s.shape[2] 
    pzl1_s = np.zeros((M0, 1, S0))
        
    for s in range(S0): # Have to retake the function for DGMM to parallelize or use apply along axis
        pzl1_s[:,:, s] = mvnorm.pdf(zl1_s[:,:,s], mean = mu_l1_s[s].flatten(order = 'C'), \
                                           cov = sigma_l1_s[s])[..., n_axis]            
    # Compute p(y | s_i = 1)
    norm_cste = np.sum(pzl1_s, axis = 0, keepdims = True) 
    norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 
 
    pzl1_s_norm = pzl1_s / norm_cste
    py_s = (pzl1_s_norm * py_zl1).sum(axis = 0)
    
    # Compute p(z |y, s) and normalize it
    norm_cste = py_s[n_axis]
    norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 
    pzl1_ys = pzl1_s * py_zl1 / norm_cste
    
    norm_cste = np.sum(pzl1_ys, axis = 0, keepdims = True) 
    norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste)     
    pzl1_ys = pzl1_ys / norm_cste

    # Compute unormalized (18)
    ps_y = w_s[n_axis] * py_s

    norm_cste = np.sum(ps_y, axis = 1, keepdims = True)     
    norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste)     

    ps_y = ps_y / norm_cste  
    p_y = py_s @ w_s[..., n_axis]
     
    return pzl1_ys, ps_y, p_y

#=============================================================================
# M Step functions
#=============================================================================

def bin_params_GLLVM(y_bin, nj_bin, lambda_bin_old, ps_y, pzl1_ys, zl1_s, AT,\
                     tol = 1E-5, maxstep = 100):
    ''' Determine the GLLVM coefficients related to binomial coefficients by 
    optimizing each column coefficients separately.
    y_bin (numobs x nb_bin nd-array): The binomial data
    nj_bin (list of int): The number of modalities for each count/binary variable
    lambda_bin_old (list of nb_ord_j x (nj_ord + r1) elements): The binomial coefficients
                                                    of the previous iteration
    ps_y ((numobs, S) nd-array): p(s | y) for all s in Omega
    pzl1_ys (nd-array): p(z1 | y, s)
    zl1_s ((M1, r1, s1) nd-array): z1 | s 
    AT ((r1 x r1) nd-array): Var(z1)^{-1/2}
    tol (int): Control when to stop the optimisation process
    maxstep (int): The maximum number of optimization step.
    ----------------------------------------------------------------------
    returns (list of nb_bin_j x (nj_ord + r1) elements): The new bin coefficients
    '''
    
    r0 = zl1_s.shape[1] 
    S0 = zl1_s.shape[2] 
    nb_bin = len(nj_bin)
    
    new_lambda_bin = []    
    
    for j in range(nb_bin):
        if j < r0 - 1: # Constrained columns
            nb_constraints = r0 - j - 1
            lcs = np.hstack([np.zeros((nb_constraints, j + 2)), np.eye(nb_constraints)])
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, 0), \
                                             np.full(nb_constraints, 0), keep_feasible = True)
        
            opt = minimize(binom_loglik_j, lambda_bin_old[j] , \
                    args = (y_bin[:,j], zl1_s, S0, ps_y, pzl1_ys, nj_bin[j]), 
                           tol = tol, method='trust-constr',  jac = bin_grad_j, \
                           constraints = linear_constraint, hess = '2-point', \
                               options = {'maxiter': maxstep})
                    
        else: # Unconstrained columns
            opt = minimize(binom_loglik_j, lambda_bin_old[j], \
                    args = (y_bin[:,j], zl1_s, S0, ps_y, pzl1_ys, nj_bin[j]), \
                           tol = tol, method='BFGS', jac = bin_grad_j, 
                           options = {'maxiter': maxstep})

        res = opt.x                
        if not(opt.success):
            res = lambda_bin_old[j]
            warnings.warn('One of the binomial optimisations has failed', RuntimeWarning)
            
        new_lambda_bin.append(deepcopy(res))  

    # Last identifiability part
    if nb_bin > 0:
        new_lambda_bin = np.stack(new_lambda_bin)
        new_lambda_bin[:,1:] = new_lambda_bin[:,1:] @ AT[0] 
        
    return new_lambda_bin



def ord_params_GLLVM(y_ord, nj_ord, lambda_ord_old, ps_y, pzl1_ys, zl1_s, AT,\
                     tol = 1E-5, maxstep = 100):
    ''' Determine the GLLVM coefficients related to ordinal coefficients by 
    optimizing each column coefficients separately.
    y_ord (numobs x nb_ord nd-array): The ordinal data
    nj_ord (list of int): The number of modalities for each ord variable
    lambda_ord_old (list of nb_ord_j x (nj_ord + r1) elements): The ordinal coefficients
                                                        of the previous iteration
    ps_y ((numobs, S) nd-array): p(s | y) for all s in Omega
    pzl1_ys (nd-array): p(z1 | y, s)
    zl1_s ((M1, r1, s1) nd-array): z1 | s 
    AT ((r1 x r1) nd-array): Var(z1)^{-1/2}
    tol (int): Control when to stop the optimisation process
    maxstep (int): The maximum number of optimization step.
    ----------------------------------------------------------------------
    returns (list of nb_ord_j x (nj_ord + r1) elements): The new ordinal coefficients
    '''
    #****************************
    # Ordinal link parameters
    #****************************  
    
    r0 = zl1_s.shape[1] 
    S0 = zl1_s.shape[2] 
    nb_ord = len(nj_ord)
    
    new_lambda_ord = []
    
    for j in range(nb_ord):
        #enc = OneHotEncoder(categories='auto')
        enc = OneHotEncoder(categories = [list(range(nj_ord[j]))])

        y_oh = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()                

        # Define the constraints such that the threshold coefficients are ordered
        nb_constraints = nj_ord[j] - 2 
        
        if nb_constraints > 0: 
            nb_params = nj_ord[j] + r0 - 1
            
            lcs = np.full(nb_constraints, -1)
            lcs = np.diag(lcs, 1)
            np.fill_diagonal(lcs, 1)
            
            lcs = np.hstack([lcs[:nb_constraints, :], \
                    np.zeros([nb_constraints, nb_params - (nb_constraints + 1)])])
            
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                np.full(nb_constraints, 0), keep_feasible = True)
                    
            opt = minimize(ord_loglik_j, lambda_ord_old[j] ,\
                    args = (y_oh, zl1_s, S0, ps_y, pzl1_ys, nj_ord[j]), 
                    tol = tol, method='trust-constr',  jac = ord_grad_j, \
                    constraints = linear_constraint, hess = '2-point',\
                        options = {'maxiter': maxstep})
                
        else: # For Nj = 2, only 2 - 1 = 1 intercept coefficient: no constraint
            opt = minimize(ord_loglik_j, lambda_ord_old[j], \
                    args = (y_oh, zl1_s, S0, ps_y, pzl1_ys, nj_ord[j]), \
                           tol = tol, method='BFGS', jac = ord_grad_j, 
                           options = {'maxiter': maxstep})
        
        res = opt.x
        if not(opt.success): # If the program fail, keep the old estimate as value
            res = lambda_ord_old[j]
            warnings.warn('One of the ordinal optimisations has failed', RuntimeWarning)
                 
        # Ensure identifiability for Lambda_j
        new_lambda_ord_j = (res[-r0: ].reshape(1, r0) @ AT[0]).flatten() 
        new_lambda_ord_j = np.hstack([deepcopy(res[: nj_ord[j] - 1]), new_lambda_ord_j]) 
        new_lambda_ord.append(new_lambda_ord_j)
    
    return new_lambda_ord
        

# This last function could be refactored with bin_params_GLLVM as they are very similar
def cont_params_GLLVM(y_cont, lambda_cont_old, ps_y, pzl1_ys, zl1_s, AT,\
                     tol = 1E-5, maxstep = 100):
    ''' Determine the GLLVM coefficients related to binomial coefficients by 
    optimizing each column coefficients separately.
    y_cont (numobs x nb_cont nd-array): The continuous data
    lambda_cont_old (list of nb_ord_j x (nj_ord + r1) elements): The continuous coefficients
                                                    of the previous iteration
    ps_y ((numobs, S) nd-array): p(s | y) for all s in Omega
    pzl1_ys (nd-array): p(z1 | y, s)
    zl1_s ((M1, r1, s1) nd-array): z1 | s 
    AT ((r1 x r1) nd-array): Var(z1)^{-1/2}
    tol (int): Control when to stop the optimisation process
    maxstep (int): The maximum number of optimization step.
    ----------------------------------------------------------------------
    returns (list of nb_bin_j x (nj_ord + r1) elements): The new bin coefficients
    '''
    
    r0 = zl1_s.shape[1] 
    S0 = zl1_s.shape[2] 
    nb_cont = y_cont.shape[1]
    
    new_lambda_cont = []    
    
    for j in range(nb_cont):
        if j < r0 - 1: # Constrained columns
            nb_constraints = r0 - j - 1
            lcs = np.hstack([np.zeros((nb_constraints, j + 2)), np.eye(nb_constraints)])
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, 0), \
                                             np.full(nb_constraints, 0), keep_feasible = True)
        
            opt = minimize(cont_loglik_j, lambda_cont_old[j] , \
                    args = (y_cont[:,j], zl1_s, S0, ps_y, pzl1_ys), 
                           tol = tol, method='trust-constr',  jac = cont_grad_j, \
                           constraints = linear_constraint, hess = '2-point', \
                               options = {'maxiter': maxstep})
                    
        else: # Unconstrained columns
            opt = minimize(cont_loglik_j, lambda_cont_old[j], \
                    args = (y_cont[:,j], zl1_s, S0, ps_y, pzl1_ys), \
                           tol = tol, method='BFGS', jac = cont_grad_j, 
                           options = {'maxiter': maxstep})

        res = opt.x                
        if not(opt.success):
            res = lambda_cont_old[j]
            warnings.warn('One of the continuous optimisations has failed', RuntimeWarning)
            
        new_lambda_cont.append(deepcopy(res))  

    # Last identifiability part
    if nb_cont > 0:
        new_lambda_cont = np.stack(new_lambda_cont)
        new_lambda_cont[:,1:] = new_lambda_cont[:,1:] @ AT[0] 
        
    return new_lambda_cont


def categ_params_GLLVM(y_categ, nj_categ, lambda_categ_old, ps_y, pzl1_ys, zl1_s, AT,\
                     tol = 1E-5, maxstep = 100):
    ''' Determine the GLLVM coefficients related to categ coefficients by 
    optimizing each column coefficients separately.
    y_categ (list of numobs x nb_categ nd-array): The categorical data
    nj_categ (list of int): The number of modalities for each categorical variable
    lambda_categ_old (list of nb_categ_j x (nj_categ + r1) elements): The categorical coefficients
                                                        of the previous iteration
    ps_y ((numobs, S) nd-array): p(s | y) for all s in Omega
    pzl1_ys (nd-array): p(z1 | y, s)
    zl1_s ((M1, r1, s1) nd-array): z1 | s 
    AT ((r1 x r1) nd-array): Var(z1)^{-1/2}
    tol (int): Control when to stop the optimisation process
    maxstep (int): The maximum number of optimization step.
    ----------------------------------------------------------------------
    returns (list of nb_ord_j x (nj_ord + r1) elements): The new ordinal coefficients
    '''
    #****************************
    # Categorical link parameters
    #****************************  
    
    r0 = zl1_s.shape[1] 
    S0 = zl1_s.shape[2] 
    nb_categ = len(nj_categ)
    
    new_lambda_categ = []
    
    for j in range(nb_categ):
        enc = OneHotEncoder(categories='auto')
        y_oh = enc.fit_transform(y_categ[:,j][..., n_axis]).toarray()    
                    
        opt = minimize(categ_loglik_j, lambda_categ_old[j], \
                    args = (y_oh, zl1_s, S0, ps_y, pzl1_ys, nj_categ[j]), \
                           tol = tol, method='BFGS', jac = categ_grad_j, 
                           options = {'maxiter': maxstep})
        
        res = opt.x
        if not(opt.success): # If the program fail, keep the old estimate as value
            res = lambda_categ_old[j]
            warnings.warn('One of the categorical optimisations has failed', RuntimeWarning)
        
        res = res.reshape(nj_categ[j], r0 + 1, order = 'C')

        # Ensure identifiability for Lambda_j
        new_lambda_categ_j = res[:, -r0: ] @ AT[0]
        new_lambda_categ_j = np.hstack([deepcopy(res[:, 0][..., n_axis]), new_lambda_categ_j]) 
        new_lambda_categ.append(new_lambda_categ_j)
    
    return new_lambda_categ