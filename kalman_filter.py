#!/usr/bin/env python
# coding: utf-8
"""
This script allows you to implement the Kalman filter. 
Written by me for NYU MSQE Computational Dynamics course. 
@author: amyrhee@nyu.edu

"""


import numpy as np
from numpy import linalg

class Kalman:
    def __init__(self, x0, sigma0, A, C, G, R, y):
        self.x0 = x0
        self.sigma0 = sigma0
        self.A = A
        self.C = C
        self.G = G
        self.R = R
        self.y = y
        
    def forecast(self):
        T = len(self.y)
        xhat = np.zeros((T,1))
        xhat[0] = self.x0
        sigmahat = np.zeros((T,1))
        sigmahat[0] = self.sigma0
        for t in range(T-1):
            a_t = self.y[t] - self.G*xhat[t]
            K_t = self.A @ sigmahat[t] @ self.G.T @ linalg.inv((self.G @ sigmahat[t] @ self.G.T) + self.R)
            
            sigmahat[t+1] = (self.A - K_t @ self.G)@sigmahat[t]@(self.A - K_t @ self.G).T + self.C@self.C.T + K_t@self.R@K_t.T
            xhat[t+1] =  self.A * xhat[t] + K_t @ a_t
        self.xhat = xhat
        self.sigmahat = sigmahat
        return self.xhat, self.sigmahat

