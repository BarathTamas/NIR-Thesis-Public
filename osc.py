# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 00:32:31 2020

@author: Tamás Baráth
based on code by Valeria Fonseca Diaz
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
# OSC
# nicomp is the number of internal components, ncomp is the number of 
 # components to remove (ncomp=1 recommended) 
class OSC:
    def __init__(self,version="SWosc",nicomp=18,ncomp=1,epsilon = 10e-6, max_iters = 20):
        self.version=version
        self.nicomp=nicomp
        self.ncomp=ncomp
        self.epsilon=epsilon
        self.max_iters=20
    def fit(self,xx,y):
        X=xx.copy()
        Y=y.copy()
        # Separating X from Y for PLS
        # Needs to be converted to numpy array from pandas df
        #X=self.df[self.freqs].to_numpy()
        # Y need to be converted to numpy array from pandas series and reshaped to (N,1) from (N,)
        #Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        
        # Self developed version
        if self.version=="SWosc":
            #Centering data
            A=np.identity(n = X.shape[0]) / X.shape[0]
            mu_x = ((A.dot(X)).sum(axis=0))/ A.sum()
            mu_y = ((A.dot(Y)).sum(axis=0))/ A.sum()
            Xc = X - mu_x 
            Yc = Y - mu_y
            
            #matrices to store loading vectors
            W = np.zeros((X.shape[1],self.ncomp))
            P = np.zeros((X.shape[1],self.ncomp))
            #setup internal PLS object
            int_pls_obj= PLSRegression(n_components=self.nicomp,scale=False)
        
        
            i = 0
            while i < self.ncomp:
                # PCA to calculate starting values
                xu, xs, xvt = np.linalg.svd(Xc)
                # starting scores
                t = xu[:,0:1]*xs[0]
                iter_i = 0
                convergence = 10 + self.epsilon
        
                while convergence > self.epsilon:
                    # orthogonalize scores
                    t_new = (np.identity(Yc.shape[0]) - Yc.dot(np.linalg.pinv(Yc.T.dot(Yc)).dot(Yc.T))).dot(t)
                    # calculate loadings by NIPALS
                    int_pls_obj.fit(Xc, t_new)
                    w=int_pls_obj.coef_
                    t=Xc.dot(w)
                    # check convergence
                    convergence = np.linalg.norm(t_new - t, axis = 0) / np.linalg.norm(t_new, axis = 0)
                    iter_i += 1
                    if iter_i > self.max_iters:
                        # print("Did not converge!")
                        convergence = 0
                    
                # After convergence calculate final loadings by regressing Xc on t
                p=Xc.T.dot(t)/(t.T.dot(t))
                # Store component's w and p
                W[:,i] = w[:,0]
                P[:,i] = p[:,0]
                # Remove component from Xc
                Xc=Xc-t.dot(p.T)
                i=i+1
            self.mu_x=mu_x
            self.X_osc=Xc
            self.W=w
            self.P=P
            
            return (Xc,W,P,mu_x)
        
        # JS osc algo courtesy of Valeria Fonseca Diaz
        # this option is giving wrong result atm, do not use until fixed!
        elif self.version=="JSosc":       
        
            X = xx.copy()
            Y = y.copy()
            
            N = X.shape[0]
            K = X.shape[1]
            q = Y.shape[1] 
        
                   
            A = np.identity(n = N) / N # this is here temporarily to include sample weights
            
            mu_x = ((A.dot(X)).sum(axis=0))/ A.sum()
            mu_y = ((A.dot(Y)).sum(axis=0))/ A.sum()
        
            Xc = X - mu_x 
            Yc = Y - mu_y
        
            W = np.zeros((X.shape[1],self.ncomp))
            P = np.zeros((X.shape[1],self.ncomp))
            TT = np.zeros((X.shape[0],self.ncomp))
        
        
            kk = 0
            while kk < self.ncomp:        
                # --- pc of xc
        
                xu, xs, xvt = np.linalg.svd(Xc)
                tt_old = xu[:,0:1]*xs[0]
                p = xvt.T[:,0:1]
                p = np.multiply(p, np.sign(np.sum(p)))
                
                iter_i = 0
                convergence = 10 + self.epsilon
        
                while convergence > self.epsilon:
        
                    # - calculate scores
                    tt = Xc.dot(p)/(p.T.dot(p))
                    # - orthogonalize scores
                    tt_new = (np.identity(Yc.shape[0]) - Yc.dot(np.linalg.pinv(Yc.T.dot(Yc)).dot(Yc.T))).dot(tt)
                    #- update loadings
                    p_new = Xc.T.dot(tt_new)/(tt_new.T.dot(tt_new))
                    # - calculate convergence
                    convergence = np.linalg.norm(tt_new - tt_old, axis = 0) / np.linalg.norm(tt_new, axis = 0)
                    # - update scores and loadings
                    tt_old = tt_new.copy()
                    p = p_new.copy()
        
                    iter_i += 1
        
                    # - check convergence in iterations
        
                    if iter_i > self.max_iters:
                        convergence = 0
        
        
        
                # - perform regression of X and t, 5 lv by default  
                int_pls_obj= PLSRegression(n_components=self.nicomp,scale=False)
                int_pls_obj.fit(Xc, tt_new)
                w=int_pls_obj.coef_
                w = w/np.linalg.norm(w)
        
                # - calculate final component that will be removed and stored
        
                tt = X.dot(w)
                tt = (np.identity(Yc.shape[0]) - Yc.dot(np.linalg.pinv(Yc.T.dot(Yc)).dot(Yc.T))).dot(tt)
                p = Xc.T.dot(tt)/(tt.T.dot(tt))
                Xc = Xc - tt.dot(p.T)
        
                # - store component
        
                W[:,kk] = w[:,0]
                P[:,kk] = p[:,0]
                TT[:,kk] = tt[:,0]
        
                kk += 1
        
            # --- final transformation of original data
            
        
            xx_new = ((xx - mu_x).dot(np.identity(n=xx.shape[1]) - W.dot(np.linalg.inv(P.T.dot(W)).dot(P.T)))) + mu_x
        
            return (xx_new, W,P,mu_x)
        
    def transform(self,X_new,mean="estimate"):
        if mean=="training":
            Xc_new=X_new-self.mu_x
        elif mean=="estimate":
            Xc_new=X_new-np.mean(X_new,axis=0)
        for comp in range(self.W.shape[1]): 
            w=self.W[:,[comp]]
            p=self.P[:,[comp]]
            # project to t
            t=Xc_new.dot(w)
            Xc_new=Xc_new-t.dot(p.T)
        return Xc_new
            