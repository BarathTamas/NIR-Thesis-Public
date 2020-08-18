# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:28:47 2020

@author: Tamás Baráth
"""
import numpy as np
import scipy as sp

class PLSRegression:
    #PLS with SIMPLS algo as I dont trust scikit-learn
    def __init__(self,n_components,P=None,scale=False):
        self.n_components = n_components
        self.P=P
        
    def fit(self,X,Y):
        assert Y.ndim == 2, 'Y needs to be a 2D array. if there is only one column, make sure it is of shape (n,1)'

        n_components=self.n_components
        P=self.P
        
        N = X.shape[0]
        K = X.shape[1]
        q = Y.shape[1]
        
        if P is None:
            P = np.identity(n = N) / N
       
        mu_x = ((P.dot(X)).sum(axis=0))/ P.sum()
        mu_y = ((P.dot(Y)).sum(axis=0))/ P.sum()
    
    
        Xc = X - mu_x
        Yc = Y - mu_y
        self.mu_x=mu_x
        self.mu_y=mu_y
    
        
        R = np.zeros((K, n_components))  # Weights to get T components
        V = np.zeros((K, n_components))  # orthonormal base for X loadings
        S = Xc.T.dot(P).dot(Yc)  # cov matrix
        
        aa = 0

        while aa < n_components:
    
            r = S[:,:]            
    
            if q > 1:
    
                U, sval, V = sp.linalg.svd(S, full_matrices=True, compute_uv=True)  
                r = U[:, 0]
    
    
            t = Xc.dot(r)
            t.shape = (N, 1)
            t = t - ((P.dot(t)).sum(axis=0)/ P.sum())
            T_scale = np.sqrt(t.T.dot(P).dot(t))
            # Normalize
            t = t / T_scale            
            r = r / T_scale
            r.shape = (K, 1)
            p = Xc.T.dot(P).dot(t)
            v = p
            v.shape = (K, 1)
    
            if aa > 0:
                v = v - V.dot(V.T.dot(p))
    
            v = v / np.sqrt(v.T.dot(v))
            S = S - v.dot(v.T.dot(S))
    
            R[:, aa] = r[:, 0]
            V[:, aa] = v[:, 0]
    
            aa += 1
        
        T = Xc.dot(R)
        
        tcal_raw0 = np.concatenate((np.ones((X.shape[0], 1)), T), axis=1)
        wtemp = np.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))          
        
        self.R=R        
        self.b = wtemp[0,0]    
        self.BPLS = R.dot(wtemp[1:, :])
        
    def predict(self, X):

        Ypred = self.mu_y + (X - self.mu_x).dot(self.BPLS)

        return Ypred
        
        
    
    