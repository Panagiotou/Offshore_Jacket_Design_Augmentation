# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:55:47 2021

@author: rfuchs
"""

import random
from shapely.geometry import Point
from scipy.stats import multivariate_normal

import autograd.numpy as np
from scipy.stats import mode 
from gower import gower_matrix

from scipy.special import binom
from scipy.optimize import minimize
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis
from sklearn.preprocessing import OneHotEncoder
from .numeric_stability import log_1plusexp, expit, softmax_
GPU = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    
    import cupy as cp
except:
    GPU =False

if GPU:
    print("RUNNING ON GPU")
else:
    print("RUNNING ON CPU")



'''
lambda_bin = lambda_bin
z_new = z

'''

def draw_new_bin(lambda_bin, z_new, nj_bin): 
    ''' A Adapter
    
    Generates draws from p(y_j | zM, s1 = k1) of the binary/count variables
    
    lambda_bin ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    z_new (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    nj_bin_j: ...
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''

    new_nb_obs = z_new.shape[0]
    r = z_new.shape[1]
    
    # Draw one y per new z for the moment
    nb_bin = len(nj_bin)
    y_bin_new = np.full((new_nb_obs, nb_bin), np.nan)
    
    for j in range(nb_bin):
    
        # Compute the probability
        eta = z_new @ lambda_bin[j][1:][..., n_axis]
        eta = eta + lambda_bin[j][0].reshape(1, 1) # Add the constant
        pi = expit(eta)
        
        # Draw the observations
        u = np.random.uniform(size = (new_nb_obs, nj_bin[j])) # To check: work for binomials
        y_bin_new[:,j] = (pi > u).sum(1)          
        
    return y_bin_new


def draw_new_categ(lambda_categ, z_new, nj_categ):
    ''' A Adapter
    Generates draws from p(y_j | zM, s1 = k1) of the categorical variables
    
    lambda_categ (nj_categ x (r + 1) ndarray): Coefficients of the categorical distributions in the GLLVM layer
    z_new (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_categ_j (int): The number of possible values values of the jth categorical variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth categorical variable
    '''  
    epsilon = 1E-10
    
    nb_categ = len(nj_categ)
    new_nb_obs = z_new.shape[0]
    r = z_new.shape[1] 
    
    y_categ_new = np.full((new_nb_obs, nb_categ), np.nan)
  
    for j in range(nb_categ):
            
        zM_broad = np.expand_dims(np.expand_dims(z_new, 1), 2)
        lambda_categ_j_ = lambda_categ[j].reshape(nj_categ[j], r + 1, order = 'C')

        # Compute the probability
        eta = zM_broad @ lambda_categ_j_[:, 1:][n_axis,..., n_axis] # Check que l'on fait r et pas k ?
        eta = eta + lambda_categ_j_[:,0].reshape(1, nj_categ[j], 1, 1, order = 'C')  # Add the constant
        pi = softmax_(eta.astype(float), axis = 1) 
        
        # Numeric stability
        pi = np.where(pi <= 0, epsilon, pi)
        pi = np.where(pi >= 1, 1 - epsilon, pi)
        
        # Draw the observations
        pi = pi[:,:,0,0]
        cumsum_pi = np.cumsum(pi, axis = 1)
        u = np.random.uniform(size = (new_nb_obs, 1)) 
        y_categ_new[:,j] = (cumsum_pi > u).argmax(1)  

    return y_categ_new


def draw_new_ord(lambda_ord, z_new, nj_ord): 
    '''  A adapter
    Generates draws from p(y_j | zM, s1 = k1) of the ordinal variables

    lambda_ord ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    z_new (... x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    nj_ord (int): The number of possible values values of the jth ordinal variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth ordinal variable
    '''    
    r = z_new.shape[1]
    nb_ord = len(nj_ord)
    new_nb_obs = z_new.shape[0]

    y_ord_new = np.full((new_nb_obs, nb_ord), np.nan)

    for j in range(nb_ord):
        
        lambda0 = lambda_ord[j][:(nj_ord[j] - 1)]
        Lambda = lambda_ord[j][-r:]
 
        broad_lambda0 = lambda0.reshape((1, nj_ord[j] - 1))
        eta = broad_lambda0 - (z_new @ Lambda.reshape((r, 1)))
        
        gamma = expit(eta)
        #gamma = np.concatenate([np.zeros((new_nb_obs, 1)), gamma], axis = 1)
        gamma = np.concatenate([gamma, np.ones((new_nb_obs, 1))], axis = 1)
        
        # Draw the observations
        u = np.random.uniform(size = (new_nb_obs, 1)) 
        y_ord_new[:,j] = (gamma > u).argmax(1)  
           
    return y_ord_new

def draw_new_cont(lambda_cont, z_new, batch_size=1000):
    '''  A adapter 
    Generates draws from p(y_j | zM, s1 = k1) of the continuous variables
    
    y_cont_j (numobs 1darray): The subset containing only the continuous variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''
    if not GPU:
        r = z_new.shape[1]
        nb_cont = lambda_cont.shape[0]
        new_nb_obs = z_new.shape[0]

        y_cont_new = np.full((new_nb_obs, nb_cont), np.nan)
        
        for j in range(nb_cont):

            eta = z_new @ lambda_cont[j][1:].reshape(r, 1)
            eta = eta + lambda_cont[j][0].reshape(1, 1) # Add the constant
            
            y_cont_new[:,j] = np.random.multivariate_normal(mean = eta.flatten(),\
                                                        cov = np.eye(new_nb_obs).astype(float))
            
        return y_cont_new
    else:
        r = z_new.shape[1]
        nb_cont = lambda_cont.shape[0]
        new_nb_obs = z_new.shape[0]

        # Transfer data to GPU
        lambda_cont_gpu = cp.array(lambda_cont)
        z_new_gpu = cp.array(z_new)

        y_cont_new = cp.full((new_nb_obs, nb_cont), cp.nan)

        for start in range(0, new_nb_obs, batch_size):
            end = min(start + batch_size, new_nb_obs)

            z_batch_gpu = z_new_gpu[start:end]
            y_batch_gpu = cp.zeros((end - start, nb_cont), dtype=cp.float32)

            for j in range(nb_cont):
                eta_gpu = z_batch_gpu @ lambda_cont_gpu[j][1:].reshape(r, 1)
                eta_gpu += lambda_cont_gpu[j][0]

                covariance_matrix_gpu = cp.eye(end - start)
                random_samples_gpu = cp.random.multivariate_normal(
                    mean=eta_gpu.flatten(),
                    cov=covariance_matrix_gpu
                )

                y_batch_gpu[:, j] = random_samples_gpu

            y_cont_new[start:end] = y_batch_gpu

        # Transfer result back to CPU
        y_cont_new_cpu = cp.asnumpy(y_cont_new)

        return y_cont_new_cpu

# def draw_new_cont(lambda_cont, z_new):
#     r = z_new.shape[1]
#     nb_cont = lambda_cont.shape[0]
#     new_nb_obs = z_new.shape[0]

#     # Transfer data to GPU
#     lambda_cont_gpu = cp.array(lambda_cont)
#     z_new_gpu = cp.array(z_new)

#     y_cont_new = cp.full((new_nb_obs, nb_cont), cp.nan)

#     for j in range(nb_cont):
#         eta_gpu = z_new_gpu @ lambda_cont_gpu[j][1:].reshape(r, 1)
#         eta_gpu += lambda_cont_gpu[j][0]

#         covariance_matrix_gpu = cp.eye(new_nb_obs)
#         random_samples_gpu = cp.random.multivariate_normal(
#             mean=eta_gpu.flatten(),
#             cov=covariance_matrix_gpu
#         )

#         y_cont_new[:, j] = random_samples_gpu

#     # Transfer result back to CPU
#     y_cont_new_cpu = cp.asnumpy(y_cont_new)

#     return y_cont_new_cpu


#========================================================
# Pi per variable (beta)
#========================================================

def stat_cont(lambda_cont, z_new):
    r = z_new.shape[1]
    nb_cont = lambda_cont.shape[0]
    
    #eta = np.full((nb_cont), np.nan)
    eta = []
    
    for j in range(nb_cont):

        eta_j = z_new @ lambda_cont[j][1:].reshape(r, 1)
        eta_j = eta_j + lambda_cont[j][0].reshape(1, 1) # Add the constant
        
        eta.append(eta_j[0][0])
        
    eta = np.array(eta)
        
    return eta

def stat_bin(lambda_bin, z_new, nj_bin):
    
    nb_bin = len(nj_bin)    
    pi = []

    for j in range(nb_bin):
    
        # Compute the probability
        eta = z_new @ lambda_bin[j][1:][..., n_axis]
        eta = eta + lambda_bin[j][0].reshape(1, 1) # Add the constant
        
        pi.append(expit(eta)[0][0])
    pi = np.array(pi)
    
    return pi * nj_bin # Return the mean


def stat_categ(lambda_categ, z_new, nj_categ):
    #epsilon = 1E-10
    nb_categ = len(nj_categ)
    r = z_new.shape[1] 
    
    pi = []
  
    for j in range(nb_categ):
        
        zM_broad = np.expand_dims(np.expand_dims(z_new, 1), 2)
        lambda_categ_j_ = lambda_categ[j].reshape(nj_categ[j], r + 1, order = 'C')

        # Compute the probability
        eta = zM_broad @ lambda_categ_j_[:, 1:][n_axis,..., n_axis] # Check que l'on fait r et pas k ?
        eta = eta + lambda_categ_j_[:,0].reshape(1, nj_categ[j], 1, 1, order = 'C')  # Add the constant
        pi.append(softmax_(eta.astype(float), axis = 1)[0,:,0,0]) 
        
    return pi

def stat_ord(lambda_ord, z_new, nj_ord):

    r = z_new.shape[1]
    nb_ord = len(nj_ord)
    new_nb_obs = z_new.shape[0]

    gamma = []

    for j in range(nb_ord):
        
        lambda0 = lambda_ord[j][:(nj_ord[j] - 1)]
        Lambda = lambda_ord[j][-r:]

        broad_lambda0 = lambda0.reshape((1, nj_ord[j] - 1))
        eta = broad_lambda0 - (z_new @ Lambda.reshape((r, 1)))
        
        gamma_j = expit(eta)

        gamma_j = np.concatenate([np.zeros((new_nb_obs, 1)), gamma_j], axis = 1)
        gamma_j = np.concatenate([gamma_j, np.ones((new_nb_obs, 1))], axis = 1)[0]

        gamma.append(gamma_j)
        

    return gamma


def stat_all(z, target, var_distrib, weights, lambda_bin, nj_bin, lambda_categ, nj_categ,\
             lambda_ord, nj_ord, lambda_cont, y_std):
     
    # Prevent the shape changes caused by the scipy minimize function
    if len(z.shape) == 1: z = z[n_axis]
    
    #=================================
    # Binary and count variables
    #=================================

    is_count = np.logical_or(var_distrib == 'binomial', var_distrib == 'bernoulli')
    count_weights = weights[is_count]   
    
    count = stat_bin(lambda_bin, z, nj_bin) 
    norm =  np.where(target[is_count] > 0, target[is_count], 1) 
    count_dist = ((count - target[is_count]) / norm) ** 2
    count_dist = np.sum(count_dist * count_weights)
    
    #=================================
    # Continuous variables
    #=================================

    cont_weights = weights[var_distrib == 'continuous']
    
    cont = stat_cont(lambda_cont, z)
    mean_cont = cont * y_std
    norm = np.where(target[var_distrib == 'continuous'] > 0,\
                     target[var_distrib == 'continuous'], 1) 
    cont_dist = ((mean_cont - target[var_distrib == 'continuous'])\
                        / norm) ** 2
    cont_dist = np.sum(cont_dist * cont_weights)

    #=================================
    # Categorical variables
    #=================================

    categ_weights = weights[var_distrib == 'categorical']
    
    nb_categ = len(nj_categ)
    categ = stat_categ(lambda_categ, z, nj_categ) 

    categ_dist = []
    for j in range(nb_categ):
        true_idx = int(target[var_distrib == 'categorical'][j]) 
        categ_dist.append((1 - categ[j][true_idx]) ** 2)

    categ_dist = np.sum(categ_dist * categ_weights)

    #=================================
    # Ordinal variables
    #=================================

    ord_weights = weights[var_distrib == 'ordinal']

    nb_ord = len(nj_ord)
    ord_ = stat_ord(lambda_ord, z, nj_ord) 
    
    ord_dist = []
    for j in range(nb_ord):
        true_idx = int(target[var_distrib == 'ordinal'][j])
        ord_dist.append((1 - (ord_[j][true_idx + 1] - ord_[j][true_idx]) ** 2))
    
    ord_dist = np.sum(ord_dist * ord_weights)

    return count_dist + categ_dist + ord_dist + cont_dist 
    

def impute(z, var_distrib, lambda_bin, nj_bin, lambda_categ, nj_categ,\
             lambda_ord, nj_ord, lambda_cont, y_std):
    
    y_bin_new = stat_bin(lambda_bin, z, nj_bin).round(0) 
    
    y_categ_all = stat_categ(lambda_categ, z[n_axis], nj_categ)
    y_categ_new = [yi.argmax() for yi in y_categ_all]
    
    y_ord_all = stat_ord(lambda_ord, z[n_axis], nj_ord)
    y_ord_new = [np.diff(yi).argmax() for yi in y_ord_all]
    
    y_cont_new = (stat_cont(lambda_cont, z[n_axis]) * y_std)[0]
       
    # Put them in the right order and append them to y
    type_counter = {'count': 0, 'ordinal': 0,\
                    'categorical': 0, 'continuous': 0} 
    
    y_new = np.full((var_distrib.shape[0]), np.nan)
    
    # Quite dirty:
    for j, var in enumerate(var_distrib):
        if (var == 'bernoulli') or (var == 'binomial'):
            y_new[j] = y_bin_new[type_counter['count']]
            type_counter['count'] =  type_counter['count'] + 1
        elif var == 'ordinal':
            y_new[j] = y_ord_new[type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        elif var == 'categorical':
            y_new[j] = y_categ_new[type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        elif var == 'continuous':
            y_new[j] = y_cont_new[type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        else:
            raise ValueError(var, 'Type not implemented')

    return y_new


#========================================================
# Optimisation process
#========================================================

def error(true, pred, cat_features, weights):
    '''
    Compute a distance between the observed values and the predicted observed
    values

    Parameters
    ----------
    true : TYPE
        DESCRIPTION.
    pred : TYPE
        DESCRIPTION.
    cat_features : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.
    

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    n_values = len(true)
    
    error = np.zeros(n_values)
    for j in range(n_values):
                
        if cat_features[j]:
            error[j] = float(true[j] != pred[j]) * weights[j]
        else:
            norm = true[j] if true[j] != 0 else 1.0
            error[j] = np.sqrt(np.mean((true[j] - pred[j]) ** 2) / norm) * weights[j]

    return error.mean()
    
def pooling(preds, var_distrib):
    p = preds.shape[1]
    pooled_pred = np.zeros(p)
    
    for j in range(p):
        if var_distrib[j] == 'categorical':
            pooled_pred[j] = str(float(mode(preds[:,j])[0][0]))
        else:
            pooled_pred[j] = preds[:,j].mean()

    return pooled_pred


def draw_obs(z, nan_mask, var_distrib, lambda_bin, lambda_ord, lambda_categ, lambda_cont,\
         nj_bin, nj_ord, nj_categ, y_std, MM):
            
    #===================================================
    # Generate a batch of pseudo-observations
    #===================================================
    
    y_bin_new = []
    y_categ_new = []
    y_ord_new = []
    y_cont_new = []
    
    for mm in range(MM):
        y_bin_new.append(draw_new_bin(lambda_bin, z[n_axis], nj_bin))
        y_categ_new.append(draw_new_categ(lambda_categ, z[n_axis], nj_categ))
        y_ord_new.append(draw_new_ord(lambda_ord, z[n_axis], nj_ord))
        y_cont_new.append(draw_new_cont(lambda_cont, z[n_axis]))
        
    # Stack the quantities
    y_bin_new = np.vstack(y_bin_new)
    y_categ_new = np.vstack(y_categ_new)
    y_ord_new = np.vstack(y_ord_new)
    y_cont_new = np.vstack(y_cont_new)
    
    # "Destandardize" the continous data
    y_cont_new = y_cont_new * y_std
    
    # Put them in the right order and append them to y
    type_counter = {'count': 0, 'ordinal': 0,\
                    'categorical': 0, 'continuous': 0} 
    
    y_new = np.full((MM, nan_mask.shape[0]), np.nan)
    
    # Quite dirty:
    for j, var in enumerate(var_distrib):
        if (var == 'bernoulli') or (var == 'binomial'):
            y_new[:, j] = y_bin_new[:, type_counter['count']]
            type_counter['count'] =  type_counter['count'] + 1
        elif var == 'ordinal':
            y_new[:, j] = y_ord_new[:, type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        elif var == 'categorical':
            y_new[:, j] = y_categ_new[:, type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        elif var == 'continuous':
            y_new[:, j] = y_cont_new[:, type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        else:
            raise ValueError(var, 'Type not implemented')

    return y_new

def dist(z, target, var_distrib, lambda_bin, lambda_ord, lambda_categ, lambda_cont,\
         nj_bin, nj_ord, nj_categ, y_std):
    
    
    MM = 100
    complete_i = np.isfinite(target)
    cat_features = var_distrib == 'categorical'
    
    y_new = draw_obs(z, ~complete_i, var_distrib, lambda_bin, lambda_ord, lambda_categ, lambda_cont,\
             nj_bin, nj_ord, nj_categ, y_std, MM)
        
    # Pool the predictions
    y_new = pooling(y_new, var_distrib)
    err = error(target[complete_i], y_new[complete_i], cat_features[complete_i])
    print(err)
    return err



from autograd import grad

def grad_stat(z, target, var_distrib, weights, lambda_bin, nj_bin, lambda_categ, nj_categ,\
             lambda_ord, nj_ord, lambda_cont, y_std):
    grad_dist = grad(stat_all)
    return grad_dist(z, target, var_distrib, weights, lambda_bin, nj_bin, lambda_categ, nj_categ,\
                 lambda_ord, nj_ord, lambda_cont, y_std)

    
#================================================
# Define convex set
#================================================


from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog

def feasible_point(A, b):
    # finds the center of the largest sphere fitting in the convex hull
    norm_vector = np.linalg.norm(A, axis=1)
    A_ = np.hstack((A, norm_vector[:, None]))
    b_ = b[:, None]
    c = np.zeros((A.shape[1] + 1,))
    c[-1] = -1
    res = linprog(c, A_ub=A_, b_ub=b[:, None], bounds=(None, None))
    return res.x[:-1]

def hs_intersection(A, b):
    interior_point = feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    return hs


def add_bbox(A, b, xrange, yrange):
    A = np.vstack((A, [
        [-1,  0],
        [ 1,  0],
        [ 0, -1],
        [ 0,  1],
    ]))
    b = np.hstack((b, [-xrange[0], xrange[1], -yrange[0], yrange[1]]))
    return A, b

def solve_convex_set(A, b, bbox, ax=None):
    A_, b_ = add_bbox(A, b, *bbox)
    interior_point = feasible_point(A_, b_)
    hs = hs_intersection(A_, b_)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs

def gen_n_point_in_polygon(n_point, polygon, tol = 0.1):
    """
    -----------
    Description
    -----------
    Generate n regular spaced points within a shapely Polygon geometry
    function from stackoverflow
    -----------
    Parameters
    -----------
    - n_point (int) : number of points required
    - polygon (shapely.geometry.polygon.Polygon) : Polygon geometry
    - tol (float) : spacing tolerance (Default is 0.1)
    -----------
    Returns
    -----------
    - points (list) : generated point geometries
    -----------
    Examples
    -----------
    >>> geom_pts = gen_n_point_in_polygon(200, polygon)
    >>> points_gs = gpd.GeoSeries(geom_pts)
    >>> points_gs.plot()
    """
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds    
    # ---- Initialize spacing and point counter
    spacing = polygon.area / n_point
    point_counter = 0
    # Start while loop to find the better spacing according to tolérance increment
    while point_counter <= n_point:
        # --- Generate grid point coordinates
        x = np.arange(np.floor(minx), int(np.ceil(maxx)), spacing)
        y = np.arange(np.floor(miny), int(np.ceil(maxy)), spacing)
        xx, yy = np.meshgrid(x,y)
        # ----
        pts = [Point(X,Y) for X,Y in zip(xx.ravel(),yy.ravel())]
        # ---- Keep only points in polygons
        points = [pt for pt in pts if pt.within(polygon)]
        # ---- Verify number of point generated
        point_counter = len(points)
        spacing -= tol
    # ---- Return
    return points




def generate_random(number, polygon):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


def fz(z, mu, sigma, w):
    '''
    Compute the density of a given z 

    Parameters
    ----------
    z : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    n_points = z.shape[0]
    K = mu.shape[0]
    pdfs = np.zeros((n_points, K))
    
    for k1 in range(K):
        pdfs[:,k1] = multivariate_normal.pdf(z, mean=mu[k1].flatten(), cov=sigma[k1])
    
    return np.sum(pdfs * w[n_axis], 1)
    

def fz(z, mu, sigma, w):
    '''
    Compute the density of a given z 

    Parameters
    ----------
    z : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    n_points = z.shape[0]
    K = mu.shape[0]
    pdfs = np.zeros((n_points, K))
    
    for k1 in range(K):
        pdfs[:,k1] = multivariate_normal.pdf(z, mean=mu[k1].flatten(), cov=sigma[k1])
    
    return np.sum(pdfs * w[n_axis], 1)


def gmm_cdf(z, mu, sigma, w):
    '''
    Compute the density of a given z 

    Parameters
    ----------
    z : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    n_points = z.shape[0]
    K = mu.shape[0]
    pdfs = np.zeros((n_points, K))
    
    for k1 in range(K):
        pdfs[:,k1] = multivariate_normal.pdf(z, mean=mu[k1].flatten(), cov=sigma[k1])
    
    return np.sum(pdfs * w[n_axis], 1)