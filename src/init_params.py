# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: RobF
"""


from copy import deepcopy
from itertools import product

from .identifiability_DGMM import identifiable_estim_DGMM, compute_z_moments,\
        diagonal_cond
from sklearn.linear_model import LogisticRegression, LinearRegression
from factor_analyzer import FactorAnalyzer
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import LabelEncoder 
from .data_preprocessing import gen_categ_as_bin_dataset, bin_to_bern
    
import prince
from .light_famd.famd import FAMD as LIGHT_FAMD
import pandas as pd

# Dirty local hard copy of the Github bevel package
from .bevel.linear_ordinal_regression import  OrderedLogit 

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis

####################################################################################
################### MCA GMM + Logistic Regressions initialisation ##################
####################################################################################

def add_missing_paths(k, init_paths, init_nb_paths):
    ''' Add the paths that have been given zeros probabily during init 
    k (dict of list): The number of components on each layer of each head and tail
    init_paths (ndarray): The already existing non-zero probability paths
    init_nb_paths (list of Bool): takes the value 1 if the path existed 0 otherwise
    ---------------------------------------------------------------------------------
    returns (tuple of size 2): The completed lists of paths (ndarray) and the total 
                                number of paths (1d array)
    '''
    
    L = len(k)
    all_possible_paths = list(product(*[np.arange(k[l]) for l in range(L)]))
    existing_paths = [tuple(path.astype(int)) for path in init_paths] # Turn them as a list of tuple
    nb_existing_paths = deepcopy(init_nb_paths)
        
    for idx, path in enumerate(all_possible_paths):
        if not(path in existing_paths):
            print('The following path has been added', idx, path)
            existing_paths.insert(idx, path)
            nb_existing_paths = np.insert(nb_existing_paths, idx, 0, axis = 0)

    return existing_paths, nb_existing_paths


def get_MFA_params(zl, kl, rl_nextl):
    ''' Determine clusters with a GMM and then adjust a Factor Model over each cluster
    zl (ndarray): The lth layer latent variable 
    kl (int): The number of components of the lth layer
    rl_nextl (1darray): The dimension of the lth layer and (l+1)th layer
    -----------------------------------------------------
    returns (dict): Dict with the parameters of the MFA approximated by GMM + FA. 
    '''
    #======================================================
    # Fit a GMM in the continuous space
    #======================================================
    numobs = zl.shape[0]
    
    not_all_groups = True
    max_trials = 100
    empty_count_counter = 0
    
    while not_all_groups:
        # If not enough obs per group then the MFA diverge...    

        gmm = GaussianMixture(n_components = kl)
        s = gmm.fit_predict(zl)
        
        clusters_found, count = np.unique(s, return_counts = True)


        if (len(clusters_found) == kl) and (count >= 2).all() :# & (count >= 5).all():
            not_all_groups = False
            
        empty_count_counter += 1
        if empty_count_counter >= max_trials:
            raise RuntimeError('Could not find a GMM init that presents the \
                               proper number of groups:', kl)
   
    
    psi = np.full((kl, rl_nextl[0], rl_nextl[0]), 0).astype(float)
    psi_inv = np.full((kl, rl_nextl[0], rl_nextl[0]), 0).astype(float)
    H = np.full((kl, rl_nextl[0], rl_nextl[1]), 0).astype(float)
    eta = np.full((kl, rl_nextl[0]), 0).astype(float)
    z_nextl = np.full((numobs, rl_nextl[1]), np.nan).astype(float)
  
    #========================================================
    # And then a MFA on each of those group
    #========================================================
    
    for j in range(kl):
        indices = (s == j)
        fa = FactorAnalyzer(rotation = None, method = 'ml', n_factors = rl_nextl[1])
        fa.fit(zl[indices])

        psi[j] = np.diag(fa.get_uniquenesses())
        H[j] = fa.loadings_
        psi_inv[j] = np.diag(1/fa.get_uniquenesses())
        z_nextl[indices] = fa.transform(zl[indices])
  
        eta[j] = np.mean(zl[indices], axis = 0)
  
    params = {'H': H, 'psi': psi, 'z_nextl': z_nextl, 'eta': eta, 'classes': s}
    return params


def dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, use_famd = False, seed = None, use_light_famd=False):
    ''' Perform dimension reduction into a continuous r dimensional space and determine 
    the init coefficients in that space
    
    y (numobs x p ndarray): The observations containing categorical variables
    n_clusters (int): The number of clusters to look for in the data
    k (1d array): The number of components of the latent Gaussian mixture layers
    r (int): The dimension of latent variables
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    var_distrib (p 1darray): An array containing the types of the variables in y 
    use_famd (Bool): Whether to the famd method (True) or not (False), to initiate the 
                    first continuous latent variable. Otherwise MCA is used.
    seed (None): The random state seed to use for the dimension reduction
    ---------------------------------------------------------------------------------------
    returns (dict): All initialisation parameters
    '''
    
    L = len(k)
    numobs = len(y)
    S = np.prod(k)
    
    #==============================================================
    # Dimension reduction performed with MCA
    #==============================================================
             
    if type(y) != pd.core.frame.DataFrame:
        raise TypeError('y should be a dataframe for prince')
    
    if (np.array(var_distrib) == 'ordinal').all():
        print('PCA init')

        pca = prince.PCA(n_components = r[0], n_iter=3, rescale_with_mean=True,\
            rescale_with_std=True, copy=True, check_input=True, engine='sklearn',\
                random_state = seed)
        z1 = pca.fit_transform(y).values

    elif use_famd:
        famd = prince.FAMD(n_components = r[0], n_iter=3, copy=True, check_input=False, \
                               engine='sklearn', random_state = seed)
        z1 = famd.fit_transform(y).values
    elif use_light_famd:
        famd = LIGHT_FAMD(n_components = r[0], n_iter=3, copy=True, check_input=False, \
                    engine='sklearn', random_state = seed)
        z1 = famd.fit_transform(y)
    else:
        # Check input = False to remove
        mca = prince.MCA(n_components = r[0], n_iter=3, copy=True,\
                         check_input=False, engine='sklearn', random_state = seed)
        z1 = mca.fit_transform(y).values
        
    z = [z1]
    y = y.values

    #==============================================================
    # Set the shape parameters of each data type
    #==============================================================    
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',\
                               var_distrib == 'binomial')].astype(int)
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',\
                              var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_ord = y[:, var_distrib == 'ordinal'].astype(float).astype(int)    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
    
    y_categ = y[:, var_distrib == 'categorical']
    nj_categ = nj[var_distrib == 'categorical']
    nb_categ = len(nj_categ)
    
    # Set y_count standard error to 1
    y_cont = y[:, var_distrib == 'continuous'] 
    
    # Before was np.float
    y_cont = y_cont / np.std(y_cont.astype(float), axis = 0, keepdims = True)
    nb_cont = y_cont.shape[1]    

    #=======================================================
    # Determining the Gaussian Parameters
    #=======================================================
    init = {}

    eta = []
    H = []
    psi = []
    paths_pred = np.zeros((numobs, L))
    
    for l in range(L):
        print(np.std(z[l],axis=0))

        params = get_MFA_params(z[l], k[l], r[l:])
        eta.append(params['eta'][..., n_axis])
        H.append(params['H'])
        psi.append(params['psi'])
        z.append(params['z_nextl']) 
        paths_pred[:, l] = params['classes']
    
    paths, nb_paths = np.unique(paths_pred, return_counts = True, axis = 0)
    paths, nb_paths = add_missing_paths(k, paths, nb_paths)
    
    w_s = nb_paths / numobs
    w_s = np.where(w_s == 0, 1E-16, w_s)
    
    # Check all paths have been explored
    if len(paths) != S:
        raise RuntimeError('Real path len is', S, 'while the initial number', \
                           'of path was only',  len(paths))

    w_s = w_s.reshape(*k).flatten('C') 

    #=============================================================
    # Enforcing identifiability constraints over the first layer
    #=============================================================
    
    H = diagonal_cond(H, psi)        
    Ez, AT = compute_z_moments(w_s, eta, H, psi)
    eta, H, psi = identifiable_estim_DGMM(eta, H, psi, Ez, AT)
    
    init['eta']  = eta     
    init['H'] = H
    init['psi'] = psi

    init['w_s'] = w_s # Probabilities of each path through the network
    init['z'] = z
    
    # The clustering layer is the one used to perform the clustering 
    # i.e. the layer l such that k[l] == n_clusters
    clustering_layer = np.argmax(np.array(k) == n_clusters)
    
    init['classes'] = paths_pred[:,clustering_layer] # 0 To change with clustering_layer_idx
     
    #=======================================================
    # Determining the coefficients of the GLLVM layer
    #=======================================================
    
    # Determining lambda_bin coefficients.
    
    lambda_bin = np.zeros((nb_bin, r[0] + 1))
    
    for j in range(nb_bin): 
        Nj = np.max(y_bin[:,j]) # The support of the jth binomial is [1, Nj]
        
        if Nj ==  1:  # If the variable is Bernoulli not binomial
            yj = y_bin[:,j]
            z_new = z[0]
        else: # If not, need to convert Binomial output to Bernoulli output
            yj, z_new = bin_to_bern(Nj, y_bin[:,j], z[0])
        
        lr = LogisticRegression()
        
        if j < r[0] - 1:
            lr.fit(z_new[:,:j + 1], yj)
            lambda_bin[j, :j + 2] = np.concatenate([lr.intercept_, lr.coef_[0]])
        else:
            lr.fit(z_new, yj)
            lambda_bin[j] = np.concatenate([lr.intercept_, lr.coef_[0]])
    
    ## Identifiability of bin coefficients
    lambda_bin[:,1:] = lambda_bin[:,1:] @ AT[0][0] 
    
    # Determining lambda_ord coefficients
    lambda_ord = []
    
    for j in range(nb_ord):
        Nj = len(np.unique(y_ord[:,j], axis = 0))  # The support of the jth ordinal is [1, Nj]
        yj = y_ord[:,j]
        
        ol = OrderedLogit()
        ol.fit(z[0], yj)
        
        ## Identifiability of ordinal coefficients
        beta_j = (ol.beta_.reshape(1, r[0]) @ AT[0][0]).flatten()
        lambda_ord_j = np.concatenate([ol.alpha_, beta_j])
        lambda_ord.append(lambda_ord_j) 
            
    # Determining the coefficients of the continuous variables
    lambda_cont = np.zeros((nb_cont, r[0] + 1))
    
    for j in range(nb_cont):
        yj = y_cont[:,j]
        linr = LinearRegression()
        
        if j < r[0] - 1:
            linr.fit(z[0][:,:j + 1], yj)
            lambda_cont[j, :j + 2] = np.concatenate([[linr.intercept_], linr.coef_])
        else:
            linr.fit(z[0], yj)
            lambda_cont[j] = np.concatenate([[linr.intercept_], linr.coef_])
    
    ## Identifiability of continuous coefficients
    lambda_cont[:,1:] = lambda_cont[:,1:] @ AT[0][0]   


    # Determining lambda_categ coefficients
    lambda_categ = []
    
    for j in range(nb_categ):
        yj = y_categ[:,j]
        
        lr = LogisticRegression(multi_class = 'multinomial')
        lr.fit(z[0], yj)        

        ## Identifiability of categ coefficients
        beta_j = lr.coef_ @ AT[0][0]   
        lambda_categ.append(np.hstack([lr.intercept_[...,n_axis], beta_j]))  
           
                
    init['lambda_bin'] = lambda_bin
    init['lambda_ord'] = lambda_ord
    init['lambda_cont'] = lambda_cont
    init['lambda_categ'] = lambda_categ
    
    if use_famd or use_light_famd:
        return init, z1, famd

    return init, None, None

