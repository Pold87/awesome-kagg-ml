# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:26:13 2015

@author: Trost
"""

import numba
import time

import numpy as np
from numba import double
from numba.decorators import jit, autojit



X= np.random.random((10000,3))


def pairwise_numpy(X):
    return np.sqrt(((X[:,None,:]-X)**2).sum(-1))



def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D= np.empty((M,M),dtype=np.float)
    
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i,k]-X[j,k]
                d +=tmp*tmp
            D[i,j] = np.sqrt(d)
    return D

pairwise_numba = autojit(pairwise_python)
    

#python
start = time.time()
    
pairwise_python(X)

end = time.time()

print(start-end)


#numpy
start = time.time()

pairwise_numpy(X)

end = time.time()

print(start-end)

#numba

start = time.time()

pairwise_numba(X)

end = time.time()

print(start-end)

#numba-numpy

