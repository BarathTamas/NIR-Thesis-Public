# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:59:25 2020

@author: Tamás Baráth
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import sys

class IntervalPLSRegression:
    def __init__(self,version="basic",ncomp=1,nint=10,cv_type="kfold",cv_param=5):
        # I have two version implemented
        # In "basic" the algorithm is extracting latent variables from the combined intervals together,
        # which I believe is the original algorithm, in this case ncomp is the max amount of LVs that can be extracted
        # In "complex" every interval gets LVs extracted separately and then an model is fit on these LVs combined,
        # in this case ncomp is the fixed number of LVs to be extracted from every interval seperately 
        # Intervals all have to have equal width (except for last ones)
        self.version=version
        self.ncomp=ncomp
        self.nint=nint
        self.cv_type=cv_type
        self.cv_param=cv_param
        
        
    def fit(self,X,y):
        self.X=X.copy()
        self.y=y.copy().to_numpy()
        # Name of predictor variables
        freqs=X.columns
        # Setup pls object
        pls = PLSRegression(n_components=self.ncomp,scale=False)
        # Check that nr of latent variables in intervals is not more than nr of variables
        max_int_width=len(X.columns)//self.nint
        if self.version=="complex" and self.ncomp > max_int_width:
            raise ValueError("Too many latent variables for number of intervals!\n{0}>{1}".format(self.ncomp,len(X.columns)//self.nint))
        # Create folds for CV used to select intervals and number of components
        if self.cv_type=="kfold":
            cv = KFold(n_splits=self.cv_param,random_state=2020)
            folds=list(cv.split(X))
        
        # Create dataframe with the intervals,
        # names of columns=interval index, the columns contain the list of variables in interval,
        # if one interval is shorter, it will have NaN in some rows
        i=0
        self.ints=pd.DataFrame()
        #Split the list of frequencies into a intervals
        for interval in np.array_split(freqs, self.nint):
            #Create a new column from the interval
            newint=pd.DataFrame(data=list(interval),columns=[i])
            #Concatenation needs to be used to handle shorter intervals
            self.ints=pd.concat([self.ints,newint], ignore_index=True, axis=1)
            i=i+1
        # Create dataframe for storing selected intervals
        self.sel_ints=pd.DataFrame()
        # Record number of latent variables used for the iterations
        self.ncomp_used=[]
        # Keep running as long as RMSE decreases or max number of intervals are reached
        RMSE_old=sys.float_info.max
        cont=True
        # For every interval the best PLS model needs to be tuned
        # Then the best interval needs to be added to the list of selected intervals
        # If it's not the first interval, the performance needs to be evaluated with the already selected ones.
        if self.version=="complex":
            while cont:
                self.cv_RMSE_all=np.zeros([len(folds),len(self.ints.columns)])
                i=0
                # Get training and validation sets
                for train,val in folds:
                    T_sel_train=np.empty([len(train), 0])
                    T_sel_val=np.empty([len(val), 0])
                    # Go over every selected interval one-by-one and get latent variables
                    for sel_interval in self.sel_ints.columns:
                        # Get variables from interval
                        sel_variables=self.sel_ints[sel_interval]
                        # Drop NaNs in case of shorter intervals
                        sel_variables=sel_variables.dropna()
                        # Convert to list for subsetting columns in pd dataframe
                        sel_variables=sel_variables.tolist()
                        # Subset the dataframe and convert to matrix
                        X_selint=self.X[sel_variables].to_numpy()
                        # Get latent variables
                        pls.fit(X_selint[train], self.y[train])
                        T_train=pls.x_scores_
                        T_val=pls.transform(X_selint[val])
                        T_sel_train=np.concatenate([T_sel_train,T_train], axis=1)
                        T_sel_val=np.concatenate([T_sel_val,T_val], axis=1)
                    j=0
                    # Get latent variables from possible new intervals,
                    # combine with selected latent components and calculate CV eror
                    for interval in self.ints.columns:
                        # Take the variables in the interval
                        variables=self.ints[interval]
                        # Drop NaNs in case of shorter intervals
                        variables=variables.dropna()
                        # Convert to list for subsetting columns in pd dataframe
                        variables=variables.tolist()
                        X_newint=self.X[variables].to_numpy()
                        # CV to check if there is improvement
                        # Inner loop is going over the cv folds
                        pls.fit(X_newint[train], self.y[train])
                        # Get latent varibles from new interval
                        T_train=pls.x_scores_
                        T_val=pls.transform(X_newint[val])
                        # Append latent variables to already selected ones column-wise
                        T_comb_train=np.concatenate([T_sel_train,T_train], axis=1)
                        T_comb_val=np.concatenate([T_sel_val,T_val], axis=1)
                        # Fit linear model
                        reg=LinearRegression().fit(T_comb_train,self.y[train])
                        self.cv_RMSE_all[i,j]=metrics.mean_squared_error(
                                self.y[val],reg.predict(T_comb_val))
                        j=j+1
                    i=i+1
                # Take the mean RMSE over the folds
                self.cv_RMSE=np.mean(self.cv_RMSE_all,axis=0)
                # Find lowest RMSE
                RMSE_new=np.amin(self.cv_RMSE)
                # Check if RMSE improved, if not stop
                if RMSE_old<=RMSE_new:
                    cont=False
                # If improved
                else:
                    RMSE_old=RMSE_new
                    # Select new interval with lowest RMSE
                    best_new_int=self.ints.columns[np.where(
                            self.cv_RMSE==RMSE_new)[0][0]]
                    # Add new interval to selected intervals
                    # Double brackets used to subset a subdataframe and not a series in order
                    # to keep column name
                    self.sel_ints=pd.concat([self.sel_ints,self.ints[[best_new_int]]], axis=1)
                    # Remove new selected interval from list of unselected ones
                    self.ints=self.ints.drop([best_new_int],axis=1)
        elif self.version=="basic":
            while cont:
                # Setup number of variables extractable
                limit=min((self.sel_ints.shape[1]+1)*max_int_width,self.ncomp)
                ncomp_range=range(1,limit+1)
                cv_RMSE_all=np.zeros([len(folds),len(self.ints.columns),len(ncomp_range)])               
                # Create list of variables already selected from the intervals
                sel_variables=[]
                for sel_int in self.sel_ints.columns:
                    # Drop NaN from shorter intervals, convert to list from pd series and append
                    sel_variables=sel_variables + self.sel_ints[sel_int].dropna().tolist()
                i=0
                for ncomp in ncomp_range:
                    pls = PLSRegression(n_components=ncomp,scale=False)
                    j=0
                    for interval in self.ints.columns:
                        # Take the variables in the interval
                        variables=self.ints[interval].dropna().tolist()
                        variables=sel_variables+variables
                        X_int=self.X[variables].to_numpy()
                        # MD cross-validation
                        k=0
                        # Inner loop is going over the cv folds
                        for train,val in folds:
                            pls.fit(X_int[train], self.y[train])
                            cv_RMSE_all[k,j,i]=metrics.mean_squared_error(
                                    self.y[val], pls.predict(X_int[val]))**0.5
                            k=k+1
                        j=j+1
                    i=i+1
                # Take the mean over the folds
                cv_RMSE=np.mean(cv_RMSE_all,axis=0)
                # Find lowest RMSE
                RMSE_new=np.amin(cv_RMSE)
                if RMSE_old<=RMSE_new:
                    cont=False
                else:
                    RMSE_old=RMSE_new
                    # Select interval with lowest RMSE
                    sel_interval=self.ints.columns[np.where(
                            cv_RMSE==RMSE_new)[0][0]]
                    # This is needed for the final model
                    self.ncomp_last=ncomp_range[np.where(
                                        cv_RMSE==RMSE_new)[1][0]]
                    # Double brackets to subset a subdataframe and not series, to keep column name
                    self.sel_ints=pd.concat([self.sel_ints,self.ints[[sel_interval]]], axis=1)
                    # Remove selected interval from list of selectable ones
                    self.ints=self.ints.drop([sel_interval],axis=1)
                if len(self.ints.columns)==0:
                    print("All variables selected!")
                    cont=False
         
        #Fit final model
        if self.version=="complex":       
            #Fit the final models on the selected intervals
            self.selints_pls_models=[]
            T_sel=np.empty([self.X.shape[0], 0])
            # Go over every selected interval one-by-one and get latent variables
            for sel_interval in self.sel_ints.columns:
                # Get variables from interval
                sel_variables=self.sel_ints[sel_interval].dropna().tolist()
                # Subset the dataframe and convert to matrix
                X_selint=self.X[sel_variables].to_numpy()
                # Get latent variables
                pls.fit(X_selint, self.y)
                # Store the PLS model of the interval
                self.selints_pls_models.append(deepcopy(pls))
                T=pls.x_scores_
                T_sel=np.concatenate([T_sel,T], axis=1)
            self.reg=LinearRegression().fit(T_sel,y)
        elif self.version=="basic":
            self.sel_variables=[]
            for sel_int in self.sel_ints.columns:
                # Drop NaN from shorter intervals, convert to list from pd series and append
                self.sel_variables=self.sel_variables + self.sel_ints[sel_int].dropna().tolist()
            X_int=self.X[self.sel_variables].to_numpy()
            self.pls_obj= PLSRegression(n_components=self.ncomp_last,scale=False)
            self.pls_obj.fit(X_int, self.y)
            
    def predict(self,X_new):
        if self.version=="complex":
            # Subset the selected intervals,
            # project the variables to latent structures in every interval
            # and then use the fitted linear regression
            TT=np.empty([X_new.shape[0], 0])
            i=0
            for interval, pls_obj in zip(self.sel_ints.columns,self.selints_pls_models):
                variables=self.sel_ints[interval].dropna().tolist()
                X=X_new[variables].to_numpy()
                T=pls_obj.transform(X)
                TT=np.concatenate([TT,T], axis=1)
                i=i+1
            return self.reg.predict(TT)
        elif self.version=="basic":
            return self.pls_obj.predict(X_new[self.sel_variables].to_numpy())


## Reading in data from Protix
#df_NIR_PX=pd.read_csv("nir_data.csv",index_col=0)
#ids_PX=df_NIR_PX['sample ID']
#
## Dropping unneccesary columns
#df_NIR_PX=df_NIR_PX.drop(["nr","sample ID"],axis=1)
## Only predicting a single target variable
#df_NIR_PX=df_NIR_PX.drop(["fat","ash","protein"],axis=1)
#df_NIR_PX.columns=df_NIR_PX.columns[:-1].tolist()+["value"]
#df_NIR_PX=df_NIR_PX.dropna(axis=0,how='any')
#
##Converting date to datetime64 from object and only keep days
#df_NIR_PX["refdate"]=pd.to_datetime(df_NIR_PX["refdate"]).dt.normalize()
#
#
##Column with dependent variable
#y_name="value"
##Date column
#date_name="refdate"
##Columns with predictors
#freqs_list = [col for col in df_NIR_PX.columns if col not in [date_name, y_name]]
##If frequency columns are not all numeric, convert them
#freqs_list=[float(freq) for freq in freqs_list]
#df_NIR_PX.columns=[float(col) if col not in [date_name, y_name] else col for col in df_NIR_PX]           
#
#obj=IntervalPLSRegression()
#obj.fit(X=df_NIR_PX[freqs_list],y=df_NIR_PX[y_name],dates=df_NIR_PX[date_name])