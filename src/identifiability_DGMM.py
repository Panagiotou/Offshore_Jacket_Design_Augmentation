# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:07:58 2020

@author: rfuchs
"""

import autograd.numpy as np
from .numeric_stability import ensure_psd
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis
from autograd.numpy.linalg import cholesky, pinv, eigh


def compute_z_moments(w_s, eta_old, H_old, psi_old):
    ''' Compute the first moment and the variance of the latent variable 
    w_s (list of length s1): The path probabilities for all s in S1
    eta_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu  
                        estimators of the previous iteration for each layer
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    psi_old (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer                        
    -------------------------------------------------------------------------
    returns (tuple of length 2): E(z^{(l)}) and Var(z^{(l)}) for all l
    '''
    
    k = [eta.shape[0] for eta in eta_old]
    L = len(eta_old) 
    
    Ez = [[] for l in range(L)]
    AT = [[] for l in range(L)]
    
    w_reshaped = w_s.reshape(*k, order = 'C')
    
    for l in reversed(range(L)):
        # Compute E(z^{(l)})
        idx_to_sum = tuple(set(range(L)) - set([l]))
        
        wl = w_reshaped.sum(idx_to_sum)[..., n_axis, n_axis]
        Ezl = (wl * eta_old[l]).sum(0, keepdims = True)
        Ez[l] = Ezl
        
        etaTeta = eta_old[l] @ t(eta_old[l], (0, 2, 1)) 
        HlHlT = H_old[l] @ t(H_old[l], (0, 2, 1)) 
        
        E_zlzlT = (wl * (HlHlT + psi_old[l] + etaTeta)).sum(0, keepdims = True)
        var_zl = E_zlzlT - Ezl @ t(Ezl, (0,2,1)) 
        var_zl = ensure_psd([var_zl])[0] # Numeric stability check
        AT_l = cholesky(var_zl)

        AT[l] = AT_l

    return Ez, AT


def identifiable_estim_DGMM(eta_old, H_old, psi_old, Ez, AT):
    ''' Enforce identifiability conditions for DGMM estimators
    eta_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu  
                        estimators of the previous iteration for each layer
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    psi_old (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer
    Ez1 (list of ndarrays): E(z^{(l)}) for all l
    AT (list of ndarrays): Var(z^{(1)})^{-1/2 T} for all l
    -------------------------------------------------------------------------
    returns (tuple of length 3): "identifiable" estimators of eta, Lambda and 
                                Psi (1st condition)
    ''' 


    L = len(eta_old)
    
    eta_new = [[] for l in range(L)]
    H_new = [[] for l in range(L)]
    psi_new = [[] for l in range(L)]
    
    for l in reversed(range(L)):
        inv_AT = pinv(AT[l])

        # Identifiability 
        psi_new[l] = inv_AT @ psi_old[l] @ t(inv_AT, (0, 2, 1))
        H_new[l] = inv_AT @ H_old[l]
        eta_new[l] = inv_AT @ (eta_old[l] -  Ez[l])    
        
    return eta_new, H_new, psi_new


def diagonal_cond(H_old, psi_old):
    ''' Ensure that Lambda^T Psi^{-1} Lambda is diagonal
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): The previous
                                        iteration values of Lambda estimators
    psi_old (list of ndarrays): The previous iteration values of Psi estimators
                    (list of nb_layers elements of shape (K_l x r_l-1, r_l-1))
    ------------------------------------------------------------------------
    returns (list of ndarrays): An "identifiable" H estimator (2nd condition)                                          
    '''
    L = len(H_old)
    
    H = []
    for l in range(L):
        B = np.transpose(H_old[l], (0, 2, 1)) @ pinv(psi_old[l]) @ H_old[l]
        values, vec  = eigh(B)
        H.append(H_old[l] @ vec)
    return H