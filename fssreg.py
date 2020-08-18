# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:29:58 2020

@author: Tamás Baráth
"""

import sys as sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


class FSSRegression:
    # Forward stepwise multiple regression based on cv error
    def __init__(self,cv_type="kfold",cval_param=5,maxvar=-1):
        self.cval_param=cval_param
        self.cv_type=cv_type
        if maxvar==-1:
            self.maxvar=sys.maxsize
        else:
            self.maxvar=maxvar
    def fit(self,X,y):
        #X should be a pandas df, Y can be anything
        self.X=X.copy()
        self.y=y.copy()
        if isinstance(self.y,pd.core.series.Series):
            self.y=self.y.to_numpy().reshape(-1,1)
        elif isinstance(self.y,np.ndarray) and self.y.shape[1]!=1:
            self.y=self.y.reshape(-1,1)
        #Some data converting.... 
        varnames=list(X.columns)
        self.bestvar=[]
        # Selecting variables based on kfold cv performance (total squared error
        # summed over all k subvalidation sets)
        if self.cv_type=="kfold":
            #Split the training data into subtraining and subvalidation sets
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
            cv_error_last=sys.maxsize
            while len(self.bestvar)<self.maxvar:
                cv_error=np.zeros(len(varnames))
                i=0
                for varname in varnames:
                    subset=self.bestvar + [varname]
                    # Subset variables
                    X_sub=self.X[subset].to_numpy()
                    # Some extra data converting...
                    if len(subset)==1:
                        X_sub=X_sub.reshape(-1,1)
                    for train, val in folds:
                        reg = LinearRegression().fit(X_sub[train], self.y[train])
                        cv_error[i]=cv_error[i] + ((reg.predict(X_sub[val])-self.y[val])**2).sum()
                    i=i+1
                if min(cv_error)>=cv_error_last:
                    break
                self.bestvar.append(varnames[cv_error.argmin()])
                varnames.remove(varnames[cv_error.argmin()])
                cv_error_last=min(cv_error)
        # Selecting variables based on R^2
        elif self.cv_type=="none":
            RSS_last=sys.maxsize
            while len(self.bestvar)<self.maxvar:
                RSSs=np.zeros(len(varnames))
                i=0
                for varname in varnames:
                    subset=self.bestvar + [varname]
                    X_sub=self.X[subset].to_numpy()
                    # Some extra data converting...
                    if len(subset)==1:
                        X_sub=X_sub.reshape(-1,1)
                    reg = LinearRegression().fit(X_sub, self.y)
                    e_sub=self.y-reg.predict(X_sub)
                    RSSs[i]=(e_sub**2).sum()
                    i=i+1
                if min(RSSs)>=RSS_last:
                    break
                self.bestvar.append(varnames[RSSs.argmin()])
                varnames.remove(varnames[RSSs.argmin()])
                RSS_last=min(RSSs)
        
        # Fit final model
        self.X_sub=self.X[self.bestvar].to_numpy()
        if len(subset)==1:
            self.X_sub=self.X_sub.reshape(-1,1)
            
        self.reg = LinearRegression().fit(self.X_sub, self.y)
        
        return self
    
    def predict(self,X_new):
        X_new_sub=X_new[self.bestvar].to_numpy()
        if len(self.bestvar)==1:
            X_new_sub=X_new_sub.reshape(-1,1)
        return self.reg.predict(X_new_sub)
