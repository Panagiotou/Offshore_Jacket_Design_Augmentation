# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:58:04 2020

@author: rfuchs
"""

import sys
from copy import deepcopy
import autograd.numpy as np
from autograd.numpy.linalg import cholesky, LinAlgError
from autograd.numpy.linalg import multi_dot, eigh

def make_symm(X):
    ''' Ensures that a matric is symmetric by setting the over-diagonal 
    coefficients as the transposed under-diagonal coefficients.
    In our case, it keeps matrices robustly symmetric to rounding errors. 
    X (2d-array): A matrix 
    ----------------------------------------------------------------------
    returns (2d-array): The "symmetrized" matrix
    '''
    return np.tril(X, k = -1) + np.tril(X).T

def make_positive_definite(m, tol = None):
    ''' Computes a matrix close to the original matrix m that is positive definite.
    This function is just a transcript of R' make.positive.definite function.
    m (2d array): A matrix that is not necessary psd.
    tol (int): A tolerence level controlling how "different" the psd matrice
                can be from the original matrix
    ---------------------------------------------------------------
    returns (2d array): A psd matrix
    '''
    d = m.shape[0]
    if (m.shape[1] != d): 
        raise RuntimeError("Input matrix is not square!")
    eigvalues, eigvect = eigh(m)
    
    # Sort the eigen values
    idx = eigvalues.argsort()[::-1]   
    eigvalues = eigvalues[idx]
    eigvect = eigvect[:,idx]
            
    if (tol == None): 
        tol = d * np.max(np.abs(eigvalues)) * sys.float_info.epsilon
    delta = 2 * tol
    tau = np.maximum(0, delta - eigvalues)
    dm = multi_dot([eigvect, np.diag(tau), eigvect.T])
    return(m + dm)

def ensure_psd(mtx_list):
    ''' Checks the positive-definiteness (psd) of a list of matrix. 
    If a matrix is not psd it is replaced by a "similar" positive-definite matrix.
    mtx_list (list of 2d-array/3d-arrays): The list of matrices to check
    ---------------------------------------------------------------------
    returns (list of 2d-array/3d-arrays): A list of matrices that are all psd.
    '''
    
    L = len(mtx_list)
    for l in range(L):
        for idx, X in enumerate(mtx_list[l]):
            try:
                cholesky(X)
            except LinAlgError:
                mtx_list[l][idx] = make_positive_definite(make_symm(X), tol = 10E-5)
    return mtx_list
                
            
def log_1plusexp(eta_):
    ''' Numerically stable version np.log(1 + np.exp(eta)) 
    eta_ (nd-array): An ndarray that potentially contains high values that 
        will overflow while taking the exponential
    -----------------------------------------------------------------------
    returns (nd-array): log(1 + exp(eta_))
    '''

    eta_original = deepcopy(eta_)
    eta_ = np.where(eta_ >= np.log(sys.float_info.max), np.log(sys.float_info.max) - 1, eta_) 
    return np.where(eta_ >= 50, eta_original, np.log1p(np.exp(eta_)))
        
def expit(eta_):
    ''' Numerically stable version of 1/(1 + exp(eta_)) 
    eta_ (nd-array): An ndarray that potentially contains high absolute values 
    that will overflow while taking the exponential.
    -----------------------------------------------------------------------
    returns (nd-array): 1/(1 + exp(eta_))   
    '''
    
    max_value_handled = np.log(np.sqrt(sys.float_info.max) - 1)
    eta_ = np.where(eta_ <= - max_value_handled + 3, - max_value_handled + 3, eta_) 
    eta_ = np.where(eta_ >= max_value_handled - 3, np.log(sys.float_info.max) - 3, eta_) 

    return np.where(eta_ <= -50, np.exp(eta_), 1/(1 + np.exp(-eta_)))
  
    
def logsumexp(a, axis=None, keepdims=False):
    """Modified from scipy :
    Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        .. versionadded:: 0.15.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.
    """
    
    a_max = np.amax(a, axis=axis, keepdims=True)

    # Cutting the max if infinite
    a_max = np.where(~np.isfinite(a_max), 0, a_max)
    assert np.sum(~np.isfinite(a_max)) == 0
    
    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out

def softmax_(x, axis=None):
    r""" Softmax function from the scipy package
    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::
        softmax(x) = np.exp(x)/sum(np.exp(x))
    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.
    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.
    """

    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
