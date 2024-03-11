# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:26:18 2020

@author: RobF
"""

from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mvnorm

import autograd.numpy as np 
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis
from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import pinv

#=============================================================================
# MC Step functions
#=============================================================================

def draw_z_s(mu_s, sigma_s, eta, M):
    ''' Draw from f(z^{l} | s) for all s in Omega and return the centered and
    non-centered draws
    mu_s (list of nd-arrays): The means of the Gaussians starting at each layer
    sigma_s (list of nd-arrays): The covariance matrices of the Gaussians 
                                                        starting at each layer
    eta (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu parameters
                                                        for each layer
    M (list of int): The number of MC to draw on each layer
    -------------------------------------------------------------------------
    returns (list of ndarrays): z^{l} | s for all s in Omega and all l in L
    '''
    
    L = len(mu_s) - 1    
    r = [mu_s[l].shape[1] for l in range(L + 1)]
    S = [mu_s[l].shape[0] for l in range(L + 1)]
    
    z_s = []
    zc_s = [] # z centered (denoted c) or all l

    for l in range(L + 1): 
        zl_s = multivariate_normal(size = (M[l], 1), \
            mean = mu_s[l].flatten(order = 'C'), cov = block_diag(*sigma_s[l]))
            
        zl_s = zl_s.reshape(M[l], S[l], r[l], order = 'C')
        z_s.append(t(zl_s, (0, 2, 1)))

        if l < L: # The last layer is already centered
            eta_ = np.repeat(t(eta[l], (2, 0, 1)), S[l + 1], axis = 1)
            zc_s.append(zl_s - eta_)

    return z_s, zc_s

def draw_z2_z1s(chsi, rho, M, r):
    ''' Draw from f(z^{l+1} | z^{l}, s, Theta) 
    chsi (list of nd-arrays): The chsi parameters for all paths starting at each layer
    rho (list of ndarrays): The rho parameters (covariance matrices) for
                                    all paths starting at each layer
    M (list of int): The number of MC to draw on each layer
    r (list of int): The dimension of each layer
    ---------------------------------------------------------------------------
    returns (list of nd-arrays): z^{l+1} | z^{l}, s, Theta for all (l,s)
    '''
    
    L = len(chsi)    
    S = [chsi[l].shape[0] for l in range(L)]
    
    z2_z1s = []
    for l in range(L):
        z2_z1s_l = np.zeros((M[l + 1], M[l], S[l], r[l + 1]))    
        for s in range(S[l]):
            z2_z1s_kl = multivariate_normal(size = M[l + 1], \
                    mean = rho[l][:,s].flatten(order = 'C'), \
                    cov = block_diag(*np.repeat(chsi[l][s][n_axis], M[l], axis = 0))) 
            
            z2_z1s_l[:, :, s] = z2_z1s_kl.reshape(M[l + 1], M[l], r[l + 1], order = 'C') 
    
        z2_z1s_l = t(z2_z1s_l, (1, 0 , 2, 3))
        z2_z1s.append(z2_z1s_l)
    
    return z2_z1s

#=============================================================================
# E Step functions
#=============================================================================


def fz2_z1s(pzl1_ys, z2_z1s, chsi, rho, S):
    ''' Compute p(z^{(l)}| z^{(l-1)}, y) 
    pzl1_ys (ndarray): p(z1 |y, s)
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    chsi (list of nd-arrays): The chsi parameters for all paths starting at each layer
    rho (list of ndarrays): The rho parameters (covariance matrices) for
                                    all paths starting at each layer
    S (list of int): The number of paths starting at each layer
    -------------------------------------------------------------------------
    returns (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    '''
    epsilon = 1E-16

    L = len(z2_z1s)
    M = [z2_z1s[l].shape[0] for l in range(L)] + [z2_z1s[-1].shape[1]]
    
    pz2_z1s = [pzl1_ys]
    for l in range(L):
            pz2_z1sm = np.zeros((M[l], M[l + 1], S[l]))  
            for s in range(S[l]):
                for m in range(M[l]): 
                    pz2_z1sm[m, :, s] = mvnorm.pdf(z2_z1s[l][m,:,s], \
                                    mean = rho[l][m, s, :, 0], \
                                    cov = chsi[l][s])
            
            norm_cste = pz2_z1sm.sum(1, keepdims = True)
            norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 

            pz2_z1sm = pz2_z1sm / norm_cste
            pz2_z1sm = np.tile(pz2_z1sm, (1, 1, S[0]//S[l]))
            pz2_z1s.append(pz2_z1sm)
            
    return pz2_z1s
        
def fz_ys(pzl1_ys, pz2_z1s):
    ''' Compute p(z^{l} | y, s) in a recursive manner 
    pzl1_ys (ndarray): p(z1 |y, s)
    pz2_z1s (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    ------------------------------------------------------------
    returns (list of ndarrays): p(z^{l} | y, s)
    '''
    
    L = len(pz2_z1s) - 1
    
    pz_ys = [pzl1_ys]
    for l in range(L):
        pz_ys_l = np.expand_dims(pz_ys[l], 2)
        pz2_z1s_l = pz2_z1s[l + 1][n_axis]
        
        pz_ys.append((pz_ys_l * pz2_z1s_l).mean(1))

    return pz_ys 



def E_step_DGMM(zl1_ys, H, z_s, zc_s, z2_z1s, pz_ys, pz2_z1s, S):
    ''' Compute the expectations of the E step for all DGMM layers
    zl1_ys ((M1, numobs, r1, S1) nd-array): z^{(1)} | y, s
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda parameters
                                                                for each layer
    z_s (list of nd-arrays): zl | s^l for all s^l and all l.
    zc_s (list of nd-arrays): (zl | s^l) - eta{k_l}^{(l)} for all s^l and all l.
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    pz_ys (list of ndarrays): p(z^{l} | y, s)
    pz2_z1s (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    S (list of int): The number of paths starting at each layer
    ------------------------------------------------------------
    returns (tuple of ndarrays): E(z^{(l)} | y, s), E(z^{(l)}z^{(l+1)T} | y, s), 
            E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    '''
    
    L = len(H)
    k = [H[l].shape[0] for l in range(L)]
    
    Ez_ys = []
    E_z1z2T_ys = []
    E_z2z2T_ys = []
    EeeT_ys = []
    
    Ez_ys.append(t(np.mean(zl1_ys, axis = 0), (0, 2, 1))) 
    
    for l in range(L):
        # Broadcast the quantities to the right shape
        z1_s = z_s[l].transpose((0, 2, 1))[..., n_axis]  
        z1_s = np.tile(z1_s, (1, np.prod(k[:l]), 1, 1)) # To recheck when L > 3

        z1c_s = np.tile(zc_s[l], (1, np.prod(k[:l]), 1))
        
        z2_s =  t(z_s[l + 1], (0, 2, 1)) 
        z2_s = np.tile(z2_s, (1, S[0] // S[l + 1], 1))[..., n_axis]  
        
        pz1_ys = pz_ys[l][..., n_axis] 
        
        H_formated = np.tile(H[l], (np.prod(k[:l]), 1, 1))
        H_formated = np.repeat(H_formated, S[l + 1], axis = 0)[n_axis] 
        
        # Compute the expectations
        ### E(z^{l + 1} | z^{l}, s) = sum_M^{l + 1} z^{l + 1}  
        # with z^{l + 1} drawn from p(z^{l + 1} | z^{l}, s)
        E_z2_z1s = z2_z1s[l].mean(1)
        E_z2_z1s = np.tile(E_z2_z1s, (1, S[0] // S[l], 1))
    
        ### E(z^{l + 1}z^{l + 1}^T | z^{l}, s) = sum_{m2=1}^M2 z2_m2 @ z2_m2T     
        E_z2z2T_z1s = (z2_z1s[l][..., n_axis] @ \
                      np.expand_dims(z2_z1s[l], 3)).mean(1)  
        E_z2z2T_z1s = np.tile(E_z2z2T_z1s, (1, S[0] // S[l], 1, 1))
        
        #### E(z^{l + 1} | y, s) = integral_z^l [ p(z^l | y, s) * E(z^{l + 1} | z^l, s) ] 
        E_z2_ys_l = (pz1_ys * E_z2_z1s[n_axis]).sum(1)   
        Ez_ys.append(E_z2_ys_l)
    
        ### E(z^{l}z^{l + 1}T | y, s) = integral_z^l [ p(z^l | y, s) * z^l @ E(z^{l + 1}T | z^l, s) ] 
        E_z1z2T_ys_l = (pz1_ys[..., n_axis] * \
                           (z1_s @ np.expand_dims(E_z2_z1s, 2))[n_axis]).sum(1)
        E_z1z2T_ys.append(E_z1z2T_ys_l)
                            
        ### E(z^{l + 1}z^{l + 1}T  | y, s) = integral_z^l [ p(z^l | y, s) @ E(z^{l + 1}z^{l + 1}T  | z1, s) ] 
        E_z2z2T_ys_l = (pz1_ys[..., n_axis] * E_z2z2T_z1s[n_axis]).sum(1)
        E_z2z2T_ys.append(E_z2z2T_ys_l)
    
        ### E[((z^l - eta^l) - Lambda z^{l + 1})((z^l - eta^l) - Lambda z^{l + 1})^T | y, s]       
        pz1z2_ys = np.expand_dims(pz_ys[l], 2) * pz2_z1s[l + 1][n_axis]
        pz1z2_ys = pz1z2_ys[..., n_axis, n_axis]
                
        e = (np.expand_dims(z1c_s, 1) - t(H_formated @ z2_s, (3, 0, 1, 2)))[..., n_axis]
        eeT = e @ t(e, (0, 1, 2, 4, 3))
        EeeT_ys_l = (pz1z2_ys * eeT[n_axis]).sum((1, 2))
        EeeT_ys.append(EeeT_ys_l)
    
    return Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys


#=============================================================================
# M Step functions
#=============================================================================

def M_step_DGMM(Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys, ps_y, H_old, k):
    ''' 
    Compute the estimators of eta, Lambda and Psi for all components and all layers
    Ez_ys (list of ndarrays): E(z^{(l)} | y, s) for all (l,s)
    E_z1z2T_ys (list of ndarrays):  E(z^{(l)}z^{(l+1)T} | y, s) 
    E_z1z2T_ys (list of ndarrays):  E(z^{(l+1)}z^{(l+1)T} | y, s) 
    EeeT_ys (list of ndarrays): E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    ps_y ((numobs, S) nd-array): p(s | y) for all s in Omega
    H_old (list of ndarrays): The previous iteration values of Lambda estimators
    k (list of int): The number of component on each layer
    --------------------------------------------------------------------------
    returns (list of ndarrays): The new estimators of eta, Lambda and Psi 
                                            for all components and all layers
    '''
    epsilon = 1E-14

    L = len(E_z1z2T_ys)
    r = [Ez_ys[l].shape[2] for l in range(L + 1)]
    numobs = len(Ez_ys[0])
    
    eta = []
    H = []
    psi = []
    
    for l in range(L):
        Ez1_ys_l = Ez_ys[l].reshape(numobs, *k, r[l], order = 'C')
        Ez2_ys_l = Ez_ys[l + 1].reshape(numobs, *k, r[l + 1], order = 'C')
        E_z1z2T_ys_l = E_z1z2T_ys[l].reshape(numobs, *k, r[l], r[l + 1], order = 'C')
        E_z2z2T_ys_l = E_z2z2T_ys[l].reshape(numobs, *k, r[l + 1], r[l + 1], order = 'C')
        EeeT_ys_l = EeeT_ys[l].reshape(numobs, *k, r[l], r[l], order = 'C')
        
        # Sum all the path going through the layer
        idx_to_sum = tuple(set(range(1, L + 1)) - set([l + 1]))
        ps_yl = ps_y.reshape(numobs, *k, order = 'C').sum(idx_to_sum)[..., n_axis, n_axis]  

        # Compute common denominator    
        den = ps_yl.sum(0)
        den = np.where(den < epsilon, epsilon, den)  
        
        # eta estimator
        eta_num = Ez1_ys_l.sum(idx_to_sum)[..., n_axis] -\
            H_old[l][n_axis] @ Ez2_ys_l.sum(idx_to_sum)[..., n_axis]
        eta_new = (ps_yl * eta_num).sum(0) / den
        
        eta.append(eta_new)
    
        # Lambda estimator
        H_num = E_z1z2T_ys_l.sum(idx_to_sum) - \
            eta_new[n_axis] @ np.expand_dims(Ez2_ys_l.sum(idx_to_sum), 2)
        
        H_new = (ps_yl * H_num  @ pinv(E_z2z2T_ys_l.sum(idx_to_sum))).sum(0) / den 
        H.append(H_new)

        # Psi estimator
        psi_new = (ps_yl * EeeT_ys_l.sum(idx_to_sum)).sum(0) / den
        psi.append(psi_new)

    return eta, H, psi