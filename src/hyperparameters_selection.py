# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:30:16 2020

@author: rfuchs
"""

import autograd.numpy as np


def M_growth(it_nb, r, numobs):
    ''' Function that controls the growth rate of M through the iterations
    it_num (int): The current iteration number
    r (list of int): The dimensions of each layer
    ---------------------------------------------------------------------
    returns (1d-array of int): The number of MC points to sample on each layer
    '''
    return ((40 / np.log(numobs)) * np.max([it_nb, 5]) * np.sqrt(r)).astype(int) 
          
    
def look_for_simpler_network(it_num):
    ''' Returns whether or not a new architecture of the network have to be 
    looking for at the current iteration.
    it_num (int): The current iteration number
    -------------------------------------------------------------------------
    returns (Bool): True if a simpler architecture has to be looking for 
                    False otherwise
    '''
    if it_num in [0, 1, 7, 10]:
        return True
    else:
        return False