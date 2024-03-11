# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:26:07 2020

@author: Utilisateur
"""

import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt

from dython.nominal import theils_u, cramers_v
from dython.nominal import associations
from scipy.stats import multivariate_normal as Nd

import autograd.numpy as np
from autograd.numpy.linalg import pinv, norm
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t


##########################################################################################################
#################################### DGMM Utils ##########################################################
##########################################################################################################


def repeat_tile(x, reps, tiles):
    ''' Repeat then tile a quantity to mimic the former code logic
    reps (int): The number of times to repeat the first axis
    tiles (int): The number of times to tile the second axis
    -----------------------------------------------------------
    returns (ndarray): The repeated then tiled nd_array
    '''
    x_rep = np.repeat(x, reps, axis = 0)
    x_tile_rep = np.tile(x_rep, (tiles, 1, 1))
    return x_tile_rep
        

def compute_path_params(eta, H, psi):
    ''' Compute the gaussian parameters for each path
    H (list of nb_layers elements of shape (K_l x r_{l-1}, r_l)): Lambda 
                                                    parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_{l-1}, r_{l-1})): Psi 
                                                    parameters for each layer
    eta (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu 
                                                    parameters for each layer
    ------------------------------------------------------------------------------------------------
    returns (tuple of len 2): The updated parameters mu_s and sigma for all s in Omega
    '''
    
    #=====================================================================
    # Retrieving model parameters
    #=====================================================================
    
    L = len(H)
    k = [len(h) for h in H]
    k_aug = k + [1] # Integrating the number of components of the last layer i.e 1
    
    r1 = H[0].shape[1]
    r2_L = [h.shape[2] for h in H] # r[2:L]
    r = [r1] + r2_L # r augmented
    
    #=====================================================================
    # Initiating the parameters for all layers
    #=====================================================================
    
    mu_s = [0 for i in range(L + 1)]
    sigma_s = [0 for i in range(L + 1)]
    
    # Initialization with the parameters of the last layer
    mu_s[-1] = np.zeros((1, r[-1], 1)) # Inverser k et r plus tard
    sigma_s[-1] = np.eye(r[-1])[n_axis]
    
    #==================================================================================
    # Compute Gaussian parameters from top to bottom for each path
    #==================================================================================

    for l in reversed(range(0, L)):
        H_repeat = np.repeat(H[l], np.prod(k_aug[l + 1: ]), axis = 0)
        eta_repeat = np.repeat(eta[l], np.prod(k_aug[l + 1: ]), axis = 0)
        psi_repeat = np.repeat(psi[l], np.prod(k_aug[l + 1: ]), axis = 0)

        mu_s[l] = eta_repeat + H_repeat @ np.tile(mu_s[l + 1], (k[l], 1, 1))
        
        sigma_s[l] = H_repeat @ np.tile(sigma_s[l + 1], (k[l], 1, 1)) @ t(H_repeat, (0, 2, 1)) \
            + psi_repeat
        
    return mu_s, sigma_s


def compute_chsi(H, psi, mu_s, sigma_s):
    ''' Compute chsi as defined in equation (8) of the DGMM paper 
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                                                    parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                                                    parameters for each layer
    mu_s (list of nd-arrays): The means of the Gaussians starting at each layer
    sigma_s (list of nd-arrays): The covariance matrices of the Gaussians 
                                                    starting at each layer
    ------------------------------------------------------------------------------------------------
    returns (list of ndarray): The chsi parameters for all paths starting at each layer
    '''
    L = len(H)
    k = [len(h) for h in H]
    
    #=====================================================================
    # Initiating the parameters for all layers
    #=====================================================================
    
    # Initialization with the parameters of the last layer    
    chsi = [0 for i in range(L)]
    chsi[-1] = pinv(pinv(sigma_s[-1]) + t(H[-1], (0, 2, 1)) @ pinv(psi[-1]) @ H[-1]) 

    #==================================================================================
    # Compute chsi from top to bottom 
    #==================================================================================
        
    for l in range(L - 1):
        Ht_psi_H = t(H[l], (0, 2, 1)) @ pinv(psi[l]) @ H[l]
        Ht_psi_H = np.repeat(Ht_psi_H, np.prod(k[l + 1:]), axis = 0)
        
        sigma_next_l = np.tile(sigma_s[l + 1], (k[l], 1, 1))
        chsi[l] = pinv(pinv(sigma_next_l) + Ht_psi_H)
            
    return chsi

def compute_rho(eta, H, psi, mu_s, sigma_s, z_c, chsi):
    ''' Compute rho as defined in equation (8) of the DGMM paper 
    eta (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu 
                                                    parameters for each layer    
    H (list of nb_layers elements of shape (K_l x r_{l-1}, r_l)): Lambda 
                                                    parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_{l-1}, r_{l-1})): Psi 
                                                    parameters for each layer
    z_c (list of nd-arrays) z^{(l)} - eta^{(l)} for each layer. 
    chsi (list of nd-arrays): The chsi parameters for each layer
    -----------------------------------------------------------------------
    returns (list of ndarrays): The rho parameters (covariance matrices) 
                                    for all paths starting at each layer
    '''
    
    L = len(H)    
    rho = [0 for i in range(L)]
    k = [len(h) for h in H]
    k_aug = k + [1] 

    for l in range(0, L):
        sigma_next_l = np.tile(sigma_s[l + 1], (k[l], 1, 1))
        mu_next_l = np.tile(mu_s[l + 1], (k[l], 1, 1))

        HxPsi_inv = t(H[l], (0, 2, 1)) @ pinv(psi[l])
        HxPsi_inv = np.repeat(HxPsi_inv, np.prod(k_aug[l + 1: ]), axis = 0)

        rho[l] = chsi[l][n_axis] @ (HxPsi_inv[n_axis] @ z_c[l][..., n_axis] \
                                    + (pinv(sigma_next_l) @ mu_next_l)[n_axis])
                
    return rho

##########################################################################################################
################################# General purposes #######################################################
##########################################################################################################
   
def isnumeric(var):
    ''' Check if a variable is numeric
    var (int, str, float etc.): The variable whom type has to be tested
    ---------------------------------------------------------------------------
    returns (Bool): Whether the variable is of numeric type (True) or not (False)    
    '''
    
    is_num = False
    try:
        int(var)
        is_num = True
    except:
        pass
    return is_num


def column_correlations(df, categorical_columns, theil_u=True):
    """
    Adapted from the table_generator package
    Column-wise correlation calculation between ``dataset_a`` and ``dataset_b``.
    :param dataset_a: First DataFrame
    :param dataset_b: Second DataFrame
    :param categorical_columns: The columns containing categorical values
    :param theil_u: Whether to use Theil's U. If False, use Cramer's V.
    :return: Mean correlation between all columns.
    """
    if categorical_columns is None:
        categorical_columns = list()
    elif categorical_columns == 'all':
        categorical_columns = df.columns

    corr = pd.DataFrame(columns=df.columns, index=['correlation'])

    for column in df.columns.tolist():
        if column in categorical_columns:
            if theil_u:
                corr[column] = theils_u(df[column].sort_values(), df[column].sort_values())
            else:
                corr[column] = cramers_v(df[column].sort_values(), df[column].sort_values())
        else:
            corr[column], _ = 0.1, ""#ss.pearsonr(df[column].sort_values(), df[column].sort_values())
    corr.fillna(value=np.nan, inplace=True)
    correlation = np.mean(corr.values.flatten())
    return correlation

# TO DO: Harmonize the code with the last plotting function
def vars_contributions(df, latent_rpz, var_distrib = [], assoc_thr = 0.0,\
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None, unit_cycle = True, ax = None):
    '''
    Plot the contribution of the original variables to the latent dimensions
    constructed by the MDGMM 

    Parameters
    ----------
    df : pandas DataFrame
        The original variables.
    latent_rpz : pandas DataFrame
        The latent representation of the observations issued by the MDGMM.
    var_distrib: numpy 1d-array 
        
    assoc_thr : int, optional
        The minimal association (in absolute value) with the latent 
        dimensions for a variable to be displayed. 
        The default is 0.0.
    title : str, optional
        The title of the plot to display. The default is 'Latent representation of the observations'.
    storage : Bool or str
        The path to store the plot
    unit_circle: Bool
        Whether to plot a unit circle along with the contributions. 
        If False a circle based on the highest contribution is drawn instead
    ax: matplotlib ax
        If not None, return the plot as a subplot hosted in the ax object
        
    Returns
    -------
    corrs: The associations computed
    '''
    
    latent_dim = latent_rpz.shape[1]
    if latent_dim > 2:
        raise NotImplementedError('This function is intended for latent\
                                  representation of dimension 2 for the moment')
                                  
    if isinstance(latent_rpz, pd.DataFrame):
        latent_rpz.columns = ['Latent dimension 1', 'Latent dimension 2']
    else:
        # Format the latent representation into a pandas DataFrame
        latent_rpz = pd.DataFrame(latent_rpz, columns = ['Latent dimension 1', 'Latent dimension 2']) 
    
    # Latent representation of the variables
    corrs = np.zeros((df.shape[1], latent_rpz.shape[1]))
    
    for j1, original_col in enumerate(df.columns):
        for j2, latent_col in enumerate(latent_rpz.columns):
            old_new = pd.DataFrame(df[original_col]).join(pd.DataFrame(latent_rpz[latent_col]))
            
            # Determine the type to compute the associations
            nominal_columns = []
            if len(var_distrib) != 0:
                if (var_distrib[j1] != 'continuous') & (var_distrib[j1] != 'binomial'):
                    nominal_columns.append(original_col)
            
            assoc = associations(old_new, nominal_columns = nominal_columns,compute_only=True)['corr'].iloc[1,0]
            corrs[j1, j2] = assoc
    
    # Plot a variable factor map for the first two dimensions.
    if ax == None:
        (fig, ax) = plt.subplots(figsize=(8, 8))
        existing_ax = False
    else:
        existing_ax = True
        
    for i in range(df.shape[1]):
        
        if (np.abs(corrs[i]) > assoc_thr).all():
            ax.arrow(0,
                     0,  # Start the arrow at the origin
                     corrs[i, 0],  #0 for PC1
                     corrs[i, 1],  #1 for PC2
                     head_width=0.02,
                     head_length=0.02)
        
            ax.text(corrs[i, 0] * 1.01,
                     corrs[i, 1] * 1.01,
                     s = df.columns.values[i])
   
    # Plot cycle towards predictions
    an = np.linspace(0, 2 * np.pi, 300)

    if unit_cycle:
        ax.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    else:
        offset = norm(corrs, axis = 1).max() * 1.2
        ax.plot(offset * np.cos(an), offset * np.sin(an))  # Add a unit circle for scale
        
    plt.axis('equal')
    ax.set_xlabel('Latent dimension 1', fontsize = 16)
    ax.set_ylabel('Latent dimension 2', fontsize = 16)
    ax.set_title(title)
    
    if storage_path:
        plt.savefig(storage_path)
        
    if not(existing_ax):
        plt.show()
    
    return corrs

# Create a plotting utility file
def obs_representation(obs_classes, latent_rpz = None, title = 'Latent representation of the observations',
                       storage_path = None):
    '''
    Plot the observations in the latent space

    Parameters
    ----------
    obs_classes : numpy array or pandas DataFrame
        The classes of each observations determined by the MDGMM.
    latent_rpz : numpy array 
        The latent representation of the observations issued by the MDGMM.
    title : str, optional
        The title of the plot to display. The default is 'Latent representation of the observations'.
    storage : Bool
        The path to store the plot
        
    Returns
    -------
    None. The plot of the observations in the latent space
    '''
            
    latent_dim = latent_rpz.shape[1]

    if latent_dim > 2:
        raise NotImplementedError('This function is intended for latent\
                                  representation of dimension 2 for the moment')
        
    if isinstance(latent_rpz, pd.DataFrame):
        latent_rpz.columns = ['Latent dimension 1', 'Latent dimension 2']
    else:
        # Format the latent representation into a pandas DataFrame
        latent_rpz = pd.DataFrame(latent_rpz, columns = ['Latent dimension 1', 'Latent dimension 2']) 
    
    classes = list(set(obs_classes))
    classes.sort()

    fig = plt.figure(figsize=(8,8))

    for cluster_idx in classes:
        cluster_data = latent_rpz.loc[obs_classes == cluster_idx]
        plt.scatter(cluster_data['Latent dimension 1'], cluster_data['Latent dimension 2'],\
                    label = cluster_idx)
        
    plt.xlabel('Latent dimension 1', fontsize = 16)
    plt.ylabel('Latent dimension 2', fontsize = 16)
    plt.title(title)

    
    plt.tight_layout()
    plt.legend()
    if storage_path:
        plt.savefig(storage_path)
        
    plt.show()

# !!! Put symbols instead of cluster number
def cluster_belonging_conf(out, latent_rpz, title = 'Cluster belonging probability',
                       storage_path = None):
    '''
    Plot the observations in the latent space

    Parameters
    ----------
    obs_classes : numpy array or pandas DataFrame
        The classes of each observations determined by the MDGMM.
    latent_rpz : numpy array 
        The latent representation of the observations issued by the MDGMM.
    title : str, optional
        The title of the plot to display. The default is 'Latent representation of the observations'.
    storage : Bool
        The path to store the plot
        
    Returns
    -------
    None. The plot of the observations in the latent space
    '''
    
    numobs = len(out['classes'])
    latent_dim = latent_rpz.shape[1]

    if latent_dim > 2:
        raise NotImplementedError('This function is intended for latent\
                                  representation of dimension 2 for the moment')
        
    if isinstance(latent_rpz, pd.DataFrame):
        latent_rpz.columns = ['Latent dimension 1', 'Latent dimension 2']
    else:
        # Format the latent representation into a pandas DataFrame
        latent_rpz = pd.DataFrame(latent_rpz, columns = ['Latent dimension 1', 'Latent dimension 2']) 
    

    fig = plt.figure(figsize=(8,8))

    ss = plt.scatter(latent_rpz['Latent dimension 1'], latent_rpz['Latent dimension 2'],\
                c = out['psl_y'].max(1),  cmap = cm.viridis)
    plt.colorbar(ss)
        
    plt.xlabel('Latent dimension 1', fontsize = 16)
    plt.ylabel('Latent dimension 2', fontsize = 16)
    plt.title(title)

    for obs_idx in range(numobs):
        plt.annotate(str(out['classes'][obs_idx]), (latent_rpz.iloc[obs_idx, 0],\
                                              latent_rpz.iloc[obs_idx, 1]))
    
    plt.tight_layout()
    #plt.legend()
    if storage_path:
        plt.savefig(storage_path)
        
    plt.show()

    
def mixtureDensity(x, y, w, mu, Sigma):
    '''
    Compute the density of a Gaussian Mixture model

    Parameters
    ----------
    x : numpy 2D array
        A meshgrid - first coordinate - on which to evaluate the density.
    y : numpy 2D array
        A meshgrid - second coordinate - on which to evaluate the density.
    w : numpy 1D array
        The proportion of the different components of the mixture.
    mu : numpy array
        The means of each component.
    Sigma : numpy array
        The covariance of each component.
    Returns
    -------
    z : np.array
        The density evaluated on the (x, y) grid.
    '''
    
    K = mu.shape[0]
    pos=np.empty(x.shape + (2,))# if  x.shape is (m,n) then  pos.shape is (m,n,2)
    pos[:, :, 0] = x; pos[:, :, 1] = y 
    z=np.zeros(x.shape)
    for k in range(K):
        z=z+w[k]*Nd.pdf(pos, mean=mu[k,:], cov=Sigma[k,:, :])
    return z
    

def density_representation(out, is_3D = False, storage_path = None, weighted = True):
    '''
    Plot the density of the DGMM distribution estimated in the latent space

    Parameters
    ----------
    out : dict
        The MDGMM output
    is_3D : Bool, optional
        Whether to plot a 3D (alternative: 2D density). The default is False.
    storage : Bool
        The path to store the plot
    weighted: Bool
        Whether to use the mixture weights or just represent the clusters location
        
    Returns
    -------
    None. The density plot

    '''
    NBPOINTS = 2000
    
    #================================================
    # Fetching the Gaussian moments and observations
    #================================================

    Sigma = out['sigma'][0]
    means = out['mu'][0][:,:,0]
    w = out['best_w_s']
    xmin, ymin = out['Ez.y'].min(0) - 0.5
    xmax, ymax = out['Ez.y'].max(0) + 0.5
    
    #================================================
    # Simulate according to the mixture density
    #================================================

    xx=np.linspace(xmin, xmax, NBPOINTS)
    yy=np.linspace(ymin, ymax, NBPOINTS)
    x,y=np.meshgrid(xx,yy)
    if weighted:
        z=mixtureDensity(x, y, w,  means, Sigma)
    else:
        equal_weights = np.ones_like(w)
        z=mixtureDensity(x, y, equal_weights,  means, Sigma)

    
    #================================================
    # Plotting the density
    #================================================
    
    fig=plt.figure(figsize=(8,8))

    if is_3D == True:
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=-90)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.RdBu,
            linewidth=0, antialiased=False)
    else:
        ax = fig.gca()
        
        ax.contourf(xx, yy, z, cmap='coolwarm')
        ax.imshow(np.rot90(z), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, z, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        
    
    ax.set_xlabel('Latent dimension 1', fontsize = 16) 
    ax.set_ylabel('Latent dimension 2', fontsize = 16)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.title('Latent dimensions density')

    plt.tight_layout()

    if storage_path:
        plt.savefig(storage_path)
        
    plt.show()
