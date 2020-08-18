# -*- coding: utf-8 -*-
"""F
Created on Wed Feb 26 10:24:21 2020

@author: Tamás Baráth
"""
import sys
import math
import random
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.colors import Normalize
from scipy import signal
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
#from pls import PLSRegression #own SIMPLS based alternative to sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import warnings
from fssreg import FSSRegression
from ipls import IntervalPLSRegression
from class_mcw_pls import mcw_pls_sklearn
from osc import OSC
warnings.filterwarnings('ignore')

class InputError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class NIRData:
    def __init__(self, df, y_name="value",date_name="refdate",
                 cval="MD",cval_param=None):
        # The class takes input dataframe in the following format:
        # -it needs to be a pandas dataframe
        # -it can only have the following columns: spectra variables,
        # measurement date, single dependent variable
        # -the measurement date and dpeendent variable column's name needs to be specified
        # -the CV method needs to be defined, it supports MD and kfold, for kfold
        # the number of folds needs to be defined with cval_param 
        self.df0=df.copy()
        self.df=df.copy()
        #Column with dependent variable
        self.y_name=y_name
        #Date column
        self.date_name=date_name
        #Columns with predictors
        self.freqs = [col for col in df.columns if col not in [date_name, y_name]]
        #If frequency columns are not all numeric, convert them
        if len([x for x in self.freqs if isinstance(x, float)])<len(self.freqs):
            self.freqs=[float(freq) for freq in self.freqs]
            self.df0.columns=[float(col) if col not in [date_name, y_name] else col for col in df.columns]
            self.df.columns=[float(col) if col not in [date_name, y_name] else col for col in df.columns]
        self.cval=cval
        if cval!="MD":
            if cval_param==None:
                raise InputError("Missing cross validation parameter!")
        self.cval_param=cval_param
     
    #Changing the cross validation method without reinstantiating the class
    def set_cval(self,cval_new):
        self.cval=cval_new        

    ################# Preprocessing techniques
    # Resetting the pre-processing to the raw spectra
    def reset(self):
        self.df=self.df0.copy()
        
    # Preprocessing methods (detrending, SG filter, SNV, MSC) 
    def to_percent(self):
        f = lambda x: x/100
        a=np.vectorize(f)(self.df[self.freqs].to_numpy())
        self.df.loc[:, self.freqs]=a

    # Convert transmittance/reflectance to absorbance
    def to_absorb(self,mode="R",percent=False):
        # If source is transmittance, use mode="T", if reflectance mode="R"
        # Functions only valid if data is between 0-1 (percent=True)
        # otherwise convert the T/R values to percent
        if not percent:
            self.to_percent()
        
        if mode=="T":
            f = lambda x: math.log10(1/x)
        elif mode=="R":
            f = lambda x: ((1-x)**2)/x
        else:
            raise Exception("Invalid mode, has to be either T or R")
        a=np.vectorize(f)(self.df[self.freqs].to_numpy())
        self.df.loc[:, self.freqs]=a
    
    # Detrending     
    def detrend(self, degree=1):
        # Calculates a linear trend or a constant (the mean) for every
        # spectral line and subtracts it
        # Result is slightly different from manually implementing it!!!
        x=np.array([self.freqs]).reshape(-1,)
        Y=self.df[self.freqs].to_numpy()
        for i in range(Y.shape[0]):
            y=Y[i,:]
            fit = np.polyfit(x, y, degree)
            trend=np.polyval(fit, x)
            y=y-trend
            Y[i,:]=y
        self.df.loc[:, self.freqs]=Y
    
    # Savitzky-Golay filter  
    def sgfilter(self,window_length=13,polyorder=2,deriv=1): 
        a=signal.savgol_filter(self.df[self.freqs]
        ,window_length, polyorder, deriv, delta=1.0,axis=-1, mode='interp', cval=0.0)
        self.df[self.freqs]=a
    
    # SNV
    def snv(self): 
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(self.df[self.freqs].T)
        self.df.loc[:, self.freqs]=scaler.transform(
                self.df[self.freqs].T).T
                
    # MSC
    def msc(self):
        ref=np.mean(self.df[self.freqs],axis=0)
        X=np.matrix(self.df[self.freqs],dtype='float')
        for i in range(self.df.shape[0]):
            A=np.vstack([np.matrix(ref,dtype='float'),
                 np.ones(X.shape[1])]).T
            coef, resids, rank, s = np.linalg.lstsq(
            A,X[i,:].T)
            X[i,:]=(X[i,:]-coef[1])/coef[0]
        self.df[self.freqs]=X
        
    # OSC is supervised preprocessing, so it needs CV, for which a joint modeling step is needed
    # this method only crossvalidates using PLS, for other models use the built in osc_params
    def osc_cv(self,nicomp_range=range(10,130,10),ncomp_range=range(1,5),epsilon = 10e-6,
               max_iters = 20,model="pls",model_parameter_range=range(1,11)):
        # Separating X from Y for PLS
        # Needs to be converted to numpy array from pandas df
        X=self.df[self.freqs].to_numpy()
        # Y need to be converted to numpy array from pandas series and reshaped to (N,1) from (N,)
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        # CV based on measurement day
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        # kfold CV
        elif self.cval=="kfold":
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
        else:
            raise InputError("Invalid CV type!")

        #Matrix for cv values for all the possible parameter combinations
        cv_RMSE_all=np.zeros([len(folds),len(model_parameter_range),len(nicomp_range),len(ncomp_range)])
        i=0
        #possible internal component values for osc
        for nicomp in nicomp_range:
            j=0
            #possible removed component values for osc
            for ncomp in ncomp_range:    
                k=0
                for train, val in folds:
                    # train osc
                    osc_obj=OSC("SWosc",nicomp,ncomp,epsilon, max_iters)
                    X_osc_train, W,P,mu_x=osc_obj.fit(X[train],Y[train])
                    # apply osc on validation set
                    # mean center data, alternatively the training set's mean can be used
                    # if you think it is a better estimate by mean="training"
                    X_osc_val=osc_obj.transform(X[val],mean="estimate")
                    l=0                           
                    #possible model patrameter values for pls
                    for param in model_parameter_range:                        
                        #setup pls model
                        pls = PLSRegression(param,scale=False)
                        #train pls
                        pls.fit(X_osc_train, Y[train])
                        #predict with pls and calculate error
                        cv_RMSE_all[k,l,i,j]=metrics.mean_squared_error(
                                Y[val], pls.predict(X_osc_val))**0.5
                        l=l+1
                    k=k+1
                j=j+1
            i=i+1
            
        # Calculate mean performance across the folds
        cv_RMSE_mean=np.mean(cv_RMSE_all,axis=0)
        # Find maximum for every osc paremeter combination
        cv_RMSE=np.amax(cv_RMSE_mean, axis=0)
        cv_RPD=np.std(self.df[self.y_name])/cv_RMSE
        fig = plt.figure(figsize=(10,5))
        ax = plt.axes(projection="3d")
        # Cartesian indexing (x,y) transposes matrix indexing (i,j)
        x, y = np.meshgrid(list(ncomp_range),list(nicomp_range))
        z=cv_RPD
        ls = LightSource(200, 45)
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)
        plt.show()
        # Best model
        print("Best RMSE: ",np.amin(cv_RMSE))
        print("Best RPD: ",np.std(self.df[self.y_name])/np.amin(cv_RMSE))
        print("Number of internal components: ",nicomp_range[np.where(
                cv_RMSE==np.amin(cv_RMSE))[0][0]])
        print("Number of removed components: ",ncomp_range[np.where(
                cv_RMSE==np.amin(cv_RMSE))[1][0]])
        return cv_RMSE
        
    ############### Plotting methods
    # Plotting the current processed version of the spectra
    def plot_spectra(self, processed=True, savefig=False, *args):
        fig,ax = plt.subplots(figsize=(12, 8)) 
        if processed:
            # Plotting unprocessed spectra
            ax.plot(self.df[self.freqs].T)
        else:
            # Plotting processed spectra
            ax.plot(self.df0[self.freqs].T)
        for arg in args:
            ax.axvline(x=arg)
        if savefig:
            plt.savefig('plot_spectra.pdf')
    
    # Plotting the fitted PLS model's regression weights on the spectra        
    def plot_pls(self):
        #r=self.pls_obj.x_rotations_
        r=self.pls_obj.coef_
        fig, ax = plt.subplots(figsize=(12, 8))    
        ax.plot(self.df[self.freqs].T,c="grey",alpha=1)
        ax.pcolorfast((np.min(self.freqs),np.max(self.freqs)), ax.get_ylim(),
                      r.T,cmap='seismic',vmin=-1,vmax=1, alpha=1)
        norm = Normalize(vmin=-1, vmax=1)
        scalarmappaple = cm.ScalarMappable(norm=norm,cmap='seismic')
        scalarmappaple.set_array(r.T)
        fig.colorbar(scalarmappaple)
            
    # Plotting the fitted MCW-PLS model's sample weights for the individual spectra        
    def plot_mcw_pls(self):
        a=np.diagonal(self.mcw_pls_obj.sample_weights)
        cmap = plt.cm.get_cmap('seismic')
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(self.df[self.freqs].shape[0]):
            row=self.df[self.freqs].iloc[i]
            ax.plot(row,c=cmap(a[i]),alpha=1)
        scalarmappaple = cm.ScalarMappable(cmap=cmap)
        scalarmappaple.set_array(a)
        plt.colorbar(scalarmappaple)
        
        r=self.mcw_pls_obj.BPLS
        fig, ax = plt.subplots(figsize=(6, 4))    
        ax.plot(self.df[self.freqs].T,c="grey",alpha=1)
        ax.pcolorfast((np.min(self.freqs),np.max(self.freqs)), ax.get_ylim(),
                      r.T,cmap='seismic',vmin=-1,vmax=1, alpha=1)
        norm = Normalize(vmin=-1, vmax=1)
        scalarmappaple = cm.ScalarMappable(norm=norm,cmap='seismic')
        scalarmappaple.set_array(r.T)
        fig.colorbar(scalarmappaple)
            
    ######################### Modeling methods
    # Support vector regression       
    # For fitting a model with given parameters
    def svr_pipe(self,gam,c,eps):
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        self.svr_pipe_obj = Pipeline([('scaler', StandardScaler()),
                                             ('support vector regression',
                                              SVR(kernel="rbf",gamma=gam,C=c,epsilon=eps))])
        self.svr_pipe_obj.fit(X, Y)
    # For evaluating a model with given parameters  
    def svr_eval(self, gam,c,eps):
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        pipe = Pipeline([('scaler', StandardScaler()),
                                             ('support vector regression',
                                              SVR(kernel="rbf",gamma=gam,C=c,epsilon=eps))])
        self.eval_df=pd.DataFrame(columns = ["estimated","true"])
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        
        cv_RMSE=np.zeros(len(folds))
        i=0
        for train, val in folds:
                pipe.fit(X[train], Y[train])
                cv_RMSE[i]=metrics.mean_squared_error(
                        Y[val], pipe.predict(X[val]))**0.5
                eval_new=pd.DataFrame({'estimated': pipe.predict(X[val]).reshape((-1,)),
                          'true': Y[val].reshape((-1,))})
                self.eval_df=self.eval_df.append(eval_new, ignore_index = True)
                i=i+1
        
        
        y_true=self.eval_df["true"]
        y_est=self.eval_df["estimated"]
        print(np.std(y_true)/metrics.mean_squared_error(y_true,y_est)**0.5)
        print(np.std(y_true)/np.mean(cv_RMSE))
        residuals=y_true-y_est
        linreg = stats.linregress(y_true, y_est)
        blue='#1f77b4'
        # Observed vs predicted
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x=y_true,y=y_est)
        # Perfect prediction
        ax.plot([np.min(Y), np.max(Y)], [np.min(Y), np.max(Y)], 'k--', color = 'r',label='Perfect fit')
        # Model fit
        ax.plot(y_true, linreg.intercept + linreg.slope*y_true, blue,label='Predicted fit')
        # Text location needs to be picked manually
        #ax.text(48, 56, 'R$^2$ = %0.002f' % linreg.rvalue,color=blue)
        ax.text(93, 95, 'R$^2$ = %0.002f' % linreg.rvalue,color=blue)
        ax.set(xlabel="Observed (%)",ylabel="Predicted (%)")
        ax.legend()
        # Predicted vs residuals
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x=y_est,y=residuals)
        ax.axhline(y=np.mean(residuals), color='r', linestyle='--',label='Mean = %0.6f' % np.mean(residuals))
        ax.set(xlabel="Predicted (%)",ylabel="Residuals (%)")
        ax.legend()
        # QQ plot
        fig,ax = plt.subplots(figsize=(5, 5))
        stats.probplot(residuals,plot=ax)
        ax.get_lines()[0].set_markerfacecolor(blue)
        ax.get_lines()[0].set_markeredgecolor(blue)
        ax.get_figure().gca().set_title("")
        ax.get_figure().gca().set_ylabel("Residuals (%)")
        # Residual density plot with normal density
        normx = np.linspace(-8,8,1000)
        normy = stats.norm.pdf(normx, loc=np.mean(residuals), scale=np.std(residuals))
        fig,ax = plt.subplots(figsize=(5, 5))
        sns.distplot(residuals,norm_hist=True,ax=ax,color=blue)
        ax.plot(normx,normy,color='r')
        sns.set_style("white")
        # Sorted alphas plot
        # Get alphas
        alphas=self.svr_pipe_obj['support vector regression'].dual_coef_
        # Take abs value and sort
        alphas=abs(alphas)
        alphas=np.sort(alphas)
        # Add zero alphas
        alphas=np.vstack((np.zeros((X.shape[0]-len(alphas.T),1)),alphas.T))
        
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.plot(alphas)
        ax.set(xlabel="Sample ranking",ylabel="SV absolute α value")
    
        
        
    
    # Method for tuning an SVM regression's free parameters based on CV    
    # OSC built in option, as this preprocessing is supervised so needs to be validated at the same time         
    def svr_cv(self,gam_start=0.001,
               c_start=100,
               eps_start=0.1,
               optimization="grid",gridscale=5,non_improve_lim=10,verbose=False,
               osc_params=None):
        # Separating X from Y for PLS
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        sample_std=np.std(self.df[self.y_name])
        # CV based on measurement day
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        # kfold CV
        elif self.cval=="kfold":
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
        else:
            raise InputError("Invalid CV type!")
        
        if optimization=="none":
            cv_RMSE=np.zeros(len(folds))
            # Only use RBF kernels, also standardize data
            pipe = Pipeline([('scaler', StandardScaler()),
                                             ('support vector regression',
                                              SVR(kernel="rbf",gamma=gam_start,C=c_start,epsilon=eps_start))])
            l=0
            for train, val in folds:
                pipe.fit(X[train], Y[train])
                cv_RMSE[l]=metrics.mean_squared_error(
                        Y[val], pipe.predict(X[val]))**0.5
                l=l+1
                gam_best=gam_start
                c_best=c_start
                eps_best=eps_start
                rpd_best=sample_std/np.mean(cv_RMSE)
                
        elif optimization=="grid":
            # Create a search vector from starting values for gridsearch
            gam_list=np.linspace(gam_start/gridscale,gam_start*gridscale,10)
            c_list=np.linspace(c_start/gridscale,c_start*gridscale,10)
            eps_list=np.linspace(eps_start/gridscale,eps_start*gridscale,10)
            
            # Create list of ndarrays from parameter search vectors,
            # it will help with making the cood more tidy
            param_lists=[gam_list,c_list,eps_list]
            param_best=np.zeros(3)
            rpd_best_all=0
            non_improve=0
            
            repeat=True
            while repeat:
                # Array for storing CV errors
                cv_RMSE_all=np.zeros([len(folds),len(gam_list),len(c_list),len(eps_list)])
                # Put the CV iteration outside to save time when using OSC
                i=0
                for train, val in folds:
                    # If OSC model specified
                    if len(osc_params)==2:
                        osc=OSC(nicomp=osc_params[0],ncomp=osc_params[1])
                        osc.fit(X[train], Y[train])
                        X_train_osc=osc.X_osc
                        X_val_osc=osc.transform(X[val])
                    j=0
                    for gam in param_lists[0]:
                        k=0
                        for c in param_lists[1]:
                            l=0
                            for eps in param_lists[2]: 
                                pipe = Pipeline([('scaler', StandardScaler()),
                                                 ('support vector regression', SVR(kernel="rbf",gamma=gam,C=c,epsilon=eps))])
                                if len(osc_params)==2:
                                    pipe.fit(X_train_osc, Y[train])
                                    cv_RMSE_all[i,j,k,l]=metrics.mean_squared_error(
                                        Y[val], pipe.predict(X_val_osc))**0.5
                                else:
                                    pipe.fit(X[train], Y[train])
                                    cv_RMSE_all[i,j,k,l]=metrics.mean_squared_error(
                                            Y[val], pipe.predict(X[val]))**0.5
                                l=l+1
                            k=k+1
                        j=j+1
                    i=i+1
                cv_RMSE=np.mean(cv_RMSE_all,axis=0)
           
                # Best model
                param_best[0]=param_lists[0][np.where(
                        cv_RMSE==np.amin(cv_RMSE))[0][0]]
                param_best[1]=param_lists[1][np.where(
                        cv_RMSE==np.amin(cv_RMSE))[1][0]]
                param_best[2]=param_lists[2][np.where(
                        cv_RMSE==np.amin(cv_RMSE))[2][0]]
                rpd_best=sample_std/np.amin(cv_RMSE)
                # Check against all time best
                if rpd_best>rpd_best_all:
                    param_best_all = param_best.copy()
                    rpd_best_all=rpd_best
                else:
                    # Increase counter if there is no improvement
                    non_improve=non_improve+1
                if verbose==True:
                    print("Best RMSE: ",np.amin(cv_RMSE))
                    print("Best RPD: ",rpd_best)
                    print("Gamma: ",param_best[0])
                    print("C: ",param_best[1])
                    print("Epsilon: ",param_best[2])
                repeat=False
                for index,p in enumerate(param_best):
                    # Check if best value is in IQ range
                    if p<np.quantile(param_lists[index],0.2) or p>np.quantile(param_lists[index],0.8):
                        # If not, move the search interval based on the magnitude of the best value
                        scale=math.floor(math.log10(p))-1
                        lower=p-(10**scale)*5
                        upper=p+(10**scale)*5
                        # If best value is at the extreme of the interval expand it by a lot that way
                        if min(param_lists[index])==p:
                            lower=min(param_lists[index])/2
                        elif max(param_lists[index])==p:
                            upper=max(param_lists[index])*2
                        # Create new search vector
                        param_lists[index]=np.linspace(lower,upper,10)
                        # Repeat evaluation
                        repeat=True
                # Terminate early if no improvements in 10 iterations        
                if non_improve>non_improve_lim:
                    repeat=False
                    print("No improvement, terminate early.")
                if repeat:
                    print("new iteration")
            # Set final values to all time best
            gam_best=param_best_all[0]
            c_best=param_best_all[1]
            eps_best=param_best_all[2]
            rpd_best=rpd_best_all
       
        # Simulated annealing
        elif optimization=="sa":
                # Number of cycles
                cycles = 100
                # Trials per cycle
                trials = 100
                # Number of accepted solutions
                n_accepted = 0.0
                # Probability of accepting worse solution at the start
                p_start = 0.3
                # Probability of accepting worse solution at the end
                p_end = 0.001
                # Initial temperature
                t_start = -1.0/math.log(p_start)
                # Final temperature
                t_end = -1.0/math.log(p_end)
                # Use geometric temp reduction
                frac = (t_end/t_start)**(1.0/(cycles-1.0))
                # Starting values
                t=t_start
                dE_mean = 0.0
                gam=gam_start
                c=c_start
                eps=eps_start
                # Calculate starting cost
                cv_RMSE=np.zeros(len(folds))
                pipe = Pipeline([('scaler', StandardScaler()),
                                     ('support vector regression',
                                      SVR(kernel="rbf",gamma=gam,C=c,epsilon=eps))])
                L=0
                for train, val in folds:
                        pipe.fit(X[train], Y[train])
                        cv_RMSE[L]=metrics.mean_squared_error(
                                Y[val], pipe.predict(X[val]))**0.5
                        L=L+1
                cost=np.mean(cv_RMSE)
                rpd=sample_std/cost
                print("starting RPD:",rpd)
                # Best results
                gam_old = gam
                c_old = c
                eps_old = eps
                cost_old=cost
                rpd_old=rpd
                # All time best result
                gam_best = gam
                c_best = c
                eps_best = eps
                cost_best=cost
                rpd_best = rpd
                for i in range(cycles):
                    if verbose and i%10==0 and i>0:
                        print('Cycle: ', i ,' with Temperature: ', t)
                        print('RPD=',rpd_old,'Gamma='  ,gam_old,', C=' ,c_old,', epsilon=',eps_old)
                    for j in range(trials):
                        # Generate new trial points
                        gam = gam_old + (random.random()-0.5)*2/1000
                        c = c_old + (random.random()-0.5)*2*10
                        eps = eps_old + (random.random()-0.5)*2/100
                        # Enforce lower bounds
                        gam = max(gam,0.0000001)
                        c = max(c,0.0000001)
                        eps = max(eps,0)
                        # Calculate cost
                        cv_RMSE=np.zeros(len(folds))
                        pipe = Pipeline([('scaler', StandardScaler()),
                                             ('support vector regression',
                                              SVR(kernel="rbf",gamma=gam,C=c,epsilon=eps))])
                        L=0
                        for train, val in folds:
                                pipe.fit(X[train], Y[train])
                                cv_RMSE[L]=metrics.mean_squared_error(
                                        Y[val], pipe.predict(X[val]))**0.5
                                L=L+1
                        cost=np.mean(cv_RMSE)
                        rpd=sample_std/cost
                        dE = cost-cost_old
                        # If new cost is higher
                        if dE > 0:
                            if (i==0 and j==0): dE_mean = dE
                            # Generate probability of acceptance
                            p = math.exp(-dE/(dE_mean * t))
                            # Determine whether to accept worse point
                            if (random.random()<p):
                                accept = True
                            else:
                                accept = False
                        else:
                            # New cost is lower, automatically accept
                            accept = True
                            # Check if cost is lower than all time best
                            if cost<cost_best:
                                # If new best, store the parameters, cost and RPD
                                gam_best=gam
                                c_best=c
                                eps_best=eps
                                cost_best=cost
                                rpd_best=rpd
                        if accept==True:
                            # Update parameters, cost and RPD
                            gam_old = gam
                            c_old = c
                            eps_old = eps
                            cost_old=cost
                            rpd_old=rpd
                            # Increment number of accepted solutions
                            n_accepted = n_accepted + 1
                            # Update energy change
                            dE_mean = (dE_mean * (n_accepted-1) +  abs(dE)) / n_accepted
                    # Lower the temperature for next cycle
                    t = frac * t
                    # Return the best setting found
        else:
            raise InputError("Invalid optimization strategy!")
        return (gam_best,c_best,eps_best,rpd_best)
            
            
            
    # Method for selecting nr of PLS components based on CV
    def pls_cv(self,ncomp_range=range(1,21),plot=False,verbose=False,
               osc_params=(10,1)):
        # Separating X from Y for PLS
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        sample_std=np.std(self.df[self.y_name])
        
        # CV based on measurement day
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        # kfold CV
        elif self.cval=="kfold":
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
        else:
            raise InputError("Invalid CV type!")
        
        # Array for storing CV errors
        cv_RMSE_all=np.zeros([len(folds),len(ncomp_range)])
        i=0
        for train, val in folds:
            # If OSC model specified
            if len(osc_params)==2:
                osc=OSC(nicomp=osc_params[0],ncomp=osc_params[1])
                osc.fit(X[train], Y[train])
                X_train_osc=osc.X_osc
                X_val_osc=osc.transform(X[val])
            j=0
            for ncomp in ncomp_range:
                pls = PLSRegression(n_components=ncomp,scale=False)
                if len(osc_params)==2:
                    pls.fit(X_train_osc, Y[train])
                    cv_RMSE_all[i,j]=metrics.mean_squared_error(
                        Y[val], pls.predict(X_val_osc))**0.5
                else:
                    pls.fit(X[train], Y[train])
                    cv_RMSE_all[i,j]=metrics.mean_squared_error(
                            Y[val], pls.predict(X[val]))**0.5
                j=j+1
            i=i+1
        # Printing and plotting CV results
        cv_RMSE_ncomp=np.mean(cv_RMSE_all,axis=0)
        cv_RPD_ncomp=sample_std/cv_RMSE_ncomp
        if plot:
            fig = plt.figure(figsize=(12,8))
            plt.gca().xaxis.grid(True)
            plt.xticks(ncomp_range)
            plt.ylabel("RPD")
            plt.xlabel("Number of components")
            plt.plot(ncomp_range,cv_RPD_ncomp)
        # Best model
        rpd_best=max(cv_RPD_ncomp)
        ncomp_best=ncomp_range[cv_RMSE_ncomp.argmin()]
        if verbose:
            print("Best RMSE: ",min(cv_RMSE_ncomp))
            print("Best RPD: ",max(cv_RPD_ncomp))
            print("Number of latent components: ",ncomp_range[cv_RMSE_ncomp.argmin()])
        return (ncomp_best,rpd_best)
    
    # Method for evaluating PLS CV performance with given nr of components
    def pls_eval(self,ncomp):
        # Separating X from Y for PLS
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        self.eval_df=pd.DataFrame(columns = ["estimated","true"]) 
        if self.cval=="MD":    
            days=self.df[self.date_name].unique()
            # DataFrame for predicted and true y values    
            pls = PLSRegression(n_components=ncomp,scale=False)
            for day in days:
                val=self.df[self.date_name]==day
                train=~val
                pls.fit(X[train], Y[train])
                # sklearn output is (N,1), has to be flattened to (N,) for pandas...
                eval_new=pd.DataFrame({'estimated': pls.predict(X[val]).reshape((-1,)),
                          'true': Y[val]})
                self.eval_df=self.eval_df.append(eval_new, ignore_index = True) 
        plt.scatter(x=self.eval_df["true"],y=self.eval_df["estimated"])
        plt.ylabel("Estimated")
        plt.xlabel("True")
        plt.axhline(y=np.mean(Y)+np.std(Y), color='r', linestyle='--')
        plt.axhline(y=np.mean(Y)-np.std(Y), color='r', linestyle='--')
        plt.axhline(y=np.mean(Y), color='r', linestyle='-')
        plt.plot([np.min(Y), np.max(Y)], [np.min(Y), np.max(Y)], 'k-', color = 'b')
     
    # Method for fitting a PLS model with given nr of components    
    def pls(self,ncomp):
        # Separating X from Y for PLS
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        self.pls_obj= PLSRegression(n_components=ncomp,scale=False)
        self.pls_obj.fit(X, Y)
    
    # Method for fitting a PLS model with given nr of components     
    def mcw_pls(self,ncomp,sig,max_iter=30, R_initial=None):
        # Separating X from Y for PLS
        # Needs to be converted to numpy array from pandas df
        X=self.df[self.freqs].to_numpy()
        # Y need to be converted to numpy array from pandas series and reshaped to (N,1) from (N,)
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        self.mcw_pls_obj=mcw_pls_sklearn(n_components=ncomp, max_iter=30, R_initial=None, scale_sigma2=sig)
        self.mcw_pls_obj.fit(X, Y)
        
    def mcw_pls_eval(self,ncomp,sig,max_iter=30, R_initial=None):
        X=self.df[self.freqs].to_numpy()
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        pls = mcw_pls_sklearn(n_components=ncomp, max_iter=30, R_initial=None, scale_sigma2=sig)
        self.eval_df=pd.DataFrame(columns = ["estimated","true"])
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        
        cv_RMSE=np.zeros(len(folds))
        i=0
        for train, val in folds:
                pls.fit(X[train], Y[train])
                cv_RMSE[i]=metrics.mean_squared_error(
                        Y[val], pls.predict(X[val]))**0.5
                eval_new=pd.DataFrame({'estimated': pls.predict(X[val]).reshape((-1,)),
                          'true': Y[val].reshape((-1,))})
                self.eval_df=self.eval_df.append(eval_new, ignore_index = True)
                i=i+1
        
        
        y_true=self.eval_df["true"]
        y_est=self.eval_df["estimated"]
        print(np.std(y_true)/metrics.mean_squared_error(y_true,y_est)**0.5)
        print(np.std(y_true)/np.mean(cv_RMSE))
        residuals=y_true-y_est
        linreg = stats.linregress(y_true, y_est)
        blue='#1f77b4'
        # Observed vs predicted
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x=y_true,y=y_est)
        # Perfect prediction
        ax.plot([np.min(Y), np.max(Y)], [np.min(Y), np.max(Y)], 'k--', color = 'r',label='Perfect fit')
        # Model fit
        ax.plot(y_true, linreg.intercept + linreg.slope*y_true, blue,label='Predicted fit')
        # Text location needs to be picked manually
        ax.text(48, 56, 'R$^2$ = %0.002f' % linreg.rvalue,color=blue)
        #ax.text(93, 95, 'R$^2$ = %0.002f' % linreg.rvalue,color=blue)
        ax.set(xlabel="Observed (%)",ylabel="Predicted (%)")
        ax.legend()
        # Predicted vs residuals
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x=y_est,y=residuals)
        ax.axhline(y=np.mean(residuals), color='r', linestyle='--',label='Mean = %0.6f' % np.mean(residuals))
        ax.set(xlabel="Predicted (%)",ylabel="Residuals (%)")
        ax.legend()
        # QQ plot
        fig,ax = plt.subplots(figsize=(5, 5))
        stats.probplot(residuals,plot=ax)
        ax.get_lines()[0].set_markerfacecolor(blue)
        ax.get_lines()[0].set_markeredgecolor(blue)
        ax.get_figure().gca().set_title("")
        ax.get_figure().gca().set_ylabel("Residuals (%)")
        # Residual density plot with normal density
        normx = np.linspace(-4,4,1000)
        normy = stats.norm.pdf(normx, loc=np.mean(residuals), scale=np.std(residuals))
        fig,ax = plt.subplots(figsize=(5, 5))
        sns.distplot(residuals,norm_hist=True,ax=ax,color=blue)
        ax.plot(normx,normy,color='r')
        sns.set_style("white")
        
        
        # Get score vector and weights for the WLS fit
        T=self.mcw_pls_obj.x_scores_
        T_aug=np.concatenate((np.ones((T.shape[0], 1)), T), axis=1)
        A=self.mcw_pls_obj.sample_weights
        a=np.diag(A)
        # Calculate hat matrix
        H=T_aug.dot(np.linalg.inv(T_aug.T.dot(A).dot(T_aug))).dot(T_aug.T).dot(A)
        h=np.diag(H).reshape(-1,1)
        # Calculate betas
        B=np.linalg.inv(T_aug.T.dot(A).dot(T_aug)).dot(T_aug.T).dot(A).dot(Y)
        # Calculate residuals
        y_hat=T_aug.dot(B)
        e=Y-y_hat
        SSE=np.sum((Y-y_hat)**2)
        n=T.shape[0]
        m=len(B)
        num=(n-m-1)**0.5
        denom=SSE*(1-h)-e**2
        e_stud=e*(num/denom)**0.5

        # Plot studentized residuals
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x=range(len(e_stud)),y=e_stud)
        ax.axhline(y=2.5, color='g', linestyle='--',label='99% interval')
        ax.axhline(y=-2.5, color='g', linestyle='--')
        ax.axhline(y=2, color='r', linestyle='--',label='95% interval')
        ax.axhline(y=-2, color='r', linestyle='--')
        ax.set(xlabel="Index",ylabel="Studentized WLS residuals")
        ax.legend(bbox_to_anchor=(0.6,0.15))
        
        # Calculate dffits

        dffits=e_stud*((h/(1-h))**0.5)
        # Plot dffits diagnostic
        fig,ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x=range(len(dffits)),y=dffits)
        ax.axhline(y=2*(m/n)**0.5, color='r', linestyle='--',label='$\pm2\cdot\sqrt{4/116}$')
        ax.axhline(y=-2*(m/n)**0.5, color='r', linestyle='--')
        ax.set(xlabel="Index",ylabel="DFFITS")
        ax.legend(bbox_to_anchor=(0.9,0.9))
        
          
     
    # Method for tuning MCW PLS model pramaters based on CV
    def mcw_pls_cv(self,ncomp_range=range(1,21),sig_start=0.1,optimization="grid",
                   plot=False,verbose=True,
                   osc_params=(10,1)):
        # Separating X from Y for PLS
        # Needs to be converted to numpy array from pandas df
        X=self.df[self.freqs].to_numpy()
        # Y need to be converted to numpy array from pandas series and reshaped to (N,1) from (N,)
        Y=self.df[self.y_name].to_numpy().reshape(-1, 1)
        sample_std=np.std(self.df[self.y_name])
        
        # CV based on measurement day
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        # kfold CV
        elif self.cval=="kfold":
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
        else:
            raise InputError("Invalid CV type!")  
        
        if optimization=="grid":
            # Create a search vector from starting values for gridsearch
            sig_list=np.linspace(sig_start/10,sig_start*10,30)

            rpd_best_all=0
            non_improve=0
            
            repeat=True
            while repeat:
                # Array for storing CV errors
                cv_RMSE_all=np.zeros([len(folds),len(ncomp_range),len(sig_list)])
                i=0
                for train,val in folds:
                    # If OSC model specified
                    if len(osc_params)==2:
                        osc=OSC(nicomp=osc_params[0],ncomp=osc_params[1])
                        osc.fit(X[train], Y[train])
                        X_train_osc=osc.X_osc
                        X_val_osc=osc.transform(X[val])
                    j=0
                    for ncomp in ncomp_range:
                        k=0
                        for sig in sig_list:
                            if len(osc_params)==2:
                                pls = mcw_pls_sklearn(n_components=ncomp, max_iter=30, R_initial=None, scale_sigma2=sig)
                                pls.fit(X_train_osc, Y[train])
                                cv_RMSE_all[i,j,k]=metrics.mean_squared_error(
                                        Y[val], pls.predict(X_val_osc))**0.5
                            else:        
                                pls = mcw_pls_sklearn(n_components=ncomp, max_iter=30, R_initial=None, scale_sigma2=sig)
                                pls.fit(X[train], Y[train])
                                cv_RMSE_all[i,j,k]=metrics.mean_squared_error(
                                        Y[val], pls.predict(X[val]))**0.5
                            k=k+1
                        j=j+1
                    i=i+1

                cv_RMSE_ncomp_sigs=np.mean(cv_RMSE_all,axis=0)

                # Best model
                ncomp_best=ncomp_range[np.where(
                        cv_RMSE_ncomp_sigs==np.amin(cv_RMSE_ncomp_sigs))[0][0]]
                sig_best=sig_list[np.where(
                        cv_RMSE_ncomp_sigs==np.amin(cv_RMSE_ncomp_sigs))[1][0]]
                rpd_best=sample_std/np.amin(cv_RMSE_ncomp_sigs)
                if verbose:
                    print("Best RMSE: ",np.amin(cv_RMSE_ncomp_sigs))
                    print("Best RPD: ",rpd_best)
                    print("Number of latent components: ",ncomp_best)
                    print("Best sigma: ",sig_best)
           
                # Check against all time best
                if rpd_best>rpd_best_all:
                    ncomp_best_all = ncomp_best
                    sig_best_all = sig_best
                    rpd_best_all= rpd_best
                else:
                    # Increase counter if there is no improvement
                    non_improve=non_improve+1
                repeat=False
                # Check if best value is in IQ range
                if sig_best<np.quantile(sig_list,0.2) or sig_best>np.quantile(sig_list,0.8):
                    # If not, move the search interval based on the magnitude of the best value
                    scale=math.floor(math.log10(sig_best))-1
                    lower=sig_best-(10**scale)*5
                    upper=sig_best+(10**scale)*5
                    # If best value is at the extreme of the interval expand it by a lot that way
                    if min(sig_list)==sig_best:
                        lower=sig_best/2
                    elif max(sig_list)==sig_best:
                        upper=sig_best*2
                    # Create new search vector
                    sig_list=np.linspace(lower,upper,10)
                    # Repeat evaluation
                    repeat=True
                # Terminate early if no improvements in 10 iterations        
                if non_improve>10:
                    repeat=False
                    print("No improvement, terminate early.")
                if repeat:
                    print("new iteration")
            # Set final values to all time best
            ncomp_best=ncomp_best_all
            sig_best=sig_best_all
            rpd_best=rpd_best_all
        

        elif optimization=="simple":

            # Array for storing CV errors
            sig_list=sig_start
            cv_RMSE_all=np.zeros([len(folds),len(ncomp_range),len(sig_list)])
            
            i=0
            for ncomp in ncomp_range:
                j=0
                for sig in sig_list:
                    pls = mcw_pls_sklearn(n_components=ncomp, max_iter=30, R_initial=None, scale_sigma2=sig)
                    k=0
                    for train,val in folds:
                        pls.fit(X[train], Y[train])
                        cv_RMSE_all[k,i,j]=metrics.mean_squared_error(
                                Y[val], pls.predict(X[val]))**0.5
                        k=k+1
                    j=j+1
                i=i+1
            
            # Printing and plotting CV results
            cv_RMSE_ncomp_sigs=np.mean(cv_RMSE_all,axis=0)
            if plot:
                cv_RPD_ncomp_sigs=sample_std/cv_RMSE_ncomp_sigs
                fig = plt.figure(figsize=(10,5))
                ax = plt.axes(projection="3d")
                # Cartesian indexing (x,y) transposes matrix indexing (i,j)
                x, y = np.meshgrid(list(sig_list),list(ncomp_range))
                z=cv_RPD_ncomp_sigs
                ls = LightSource(270, 45)
                rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
                surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                                       linewidth=0, antialiased=False, shade=False)
    
            plt.show()
            # Best model
            ncomp_best=ncomp_range[np.where(
                    cv_RMSE_ncomp_sigs==np.amin(cv_RMSE_ncomp_sigs))[0][0]]
            sig_best=sig_list[np.where(
                    cv_RMSE_ncomp_sigs==np.amin(cv_RMSE_ncomp_sigs))[1][0]]
            rpd_best=sample_std/np.amin(cv_RMSE_ncomp_sigs)
            print("Best RMSE: ",np.amin(cv_RMSE_ncomp_sigs))
            print("Best RPD: ",rpd_best)
            print("Number of latent components: ",ncomp_best)
            print("Best sigma: ",sig_best)
    
        return (ncomp_best,sig_best,rpd_best)
    
    # Cross-Validation method for Interval Partial Least Squares
    def ipls_cv(self,version="basic",nint_list=[8,16,32],ncomp_range=range(1,10),
                inner_cv="kfold",inner_cv_param=5,verbose=True,
                osc_params=(10,1)):
               
        X=self.df[self.freqs]
        Y=self.df[self.y_name]

        # CV based on measurement day
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        # kfold CV
        elif self.cval=="kfold":
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
        else:
            raise InputError("Invalid CV type!") 
            
        #Array for cv values
        cv_RMSE_all=np.zeros([len(folds),len(ncomp_range),len(nint_list)])
        i=0
        for train,val in folds:
            # If OSC model specified
            if len(osc_params)==2:
                osc=OSC(nicomp=osc_params[0],ncomp=osc_params[1])
                # IPLS needs column names, so it uses pandas, but osc uses numpy arrays
                osc.fit(X.iloc[train].to_numpy(), Y.iloc[train].to_numpy().reshape(-1,1))
                X_train_osc=pd.DataFrame(data=osc.X_osc,columns=self.freqs)
                X_val_osc=pd.DataFrame(data=osc.transform(X.iloc[val].to_numpy()),columns=self.freqs)
            j=0
            for ncomp in ncomp_range:
                k=0
                for nint in nint_list:
                    ipls_obj=IntervalPLSRegression(ncomp=ncomp,nint=nint,
                                                       cv_type=inner_cv,cv_param=inner_cv_param)
                    if len(osc_params)==2:
                        ipls_obj.fit(X_train_osc, Y.iloc[train])
                        cv_RMSE_all[i,j,k]=metrics.mean_squared_error(
                                 Y.iloc[val], ipls_obj.predict(X_val_osc))**0.5
                    else:
                        ipls_obj.fit(X.iloc[train], Y.iloc[train])
                        cv_RMSE_all[i,j,k]=metrics.mean_squared_error(
                                 Y.iloc[val], ipls_obj.predict(X.iloc[val]))**0.5
                    k=k+1
                j=j+1
            i=i+1
        cv_RMSE=np.mean(cv_RMSE_all,axis=0)
        RMSE_best=np.amin(cv_RMSE)
        
        rpd_best=np.std(self.df[self.y_name])/RMSE_best
        # Best model
        ncomp_best=ncomp_range[np.where(
                cv_RMSE==RMSE_best)[0][0]]
        nint_best=nint_list[np.where(
                cv_RMSE==RMSE_best)[1][0]]
        if verbose:
            print("Best RMSE: ",RMSE_best)
            print("Best RPD: ",rpd_best)
            print("Number of components:",ncomp_best)
            print("Number of intervals:",nint_best)
        return (ncomp_best,nint_best,rpd_best)

    # Method for selecting a multiple linear regression model using
    # stepwise forward selection. The selection of variables is done using kfold CV
    #or without CV.
    # The performance of different number of selected variables is evaluated with
    # measurement day or kfold CV.
    def fssregression_cv(self,inner_cv="kfold",inner_cv_param=5,maxvar=2,verbose=False,
                         osc_params=(10,1)):
        #inner CV can be "kfold" or "none"
        # Separating X from Y for PLS
        X=self.df[self.freqs]
        Y=self.df[self.y_name]
        
        # Create list for selected variables
        best_vars=[]
        reg = FSSRegression(inner_cv,inner_cv_param,maxvar)
        # CV based on measurement day
        if self.cval=="MD":
            cv = LeaveOneGroupOut()
            folds=list(cv.split(X=X,y=Y,groups=self.df[self.date_name]))
        # kfold CV
        elif self.cval=="kfold":
            cv = KFold(n_splits=self.cval_param)
            folds=list(cv.split(X))
        else:
            raise InputError("Invalid CV type!")  
        i=0
        #Array for cv values
        cv_RMSE_all=np.zeros([len(folds)])
        for train,val in folds:
            # If OSC model specified
            if len(osc_params)==2:
                osc=OSC(nicomp=osc_params[0],ncomp=osc_params[1])
                # FSSR needs column names, so it uses pandas, but osc uses numpy arrays
                osc.fit(X.iloc[train].to_numpy(), Y.iloc[train].to_numpy().reshape(-1,1))
                X_train_osc=pd.DataFrame(data=osc.X_osc,columns=self.freqs)
                X_val_osc=pd.DataFrame(data=osc.transform(X.iloc[val].to_numpy()),columns=self.freqs)
                # Fit and predict
                reg.fit(X_train_osc, Y.iloc[train])
                cv_RMSE_all[i]=metrics.mean_squared_error(
                        Y.iloc[val], reg.predict(X_val_osc))**0.5
                best_vars.append(reg.bestvar) 
            else:
                reg.fit(X.iloc[train], Y.iloc[train])
                cv_RMSE_all[i]=metrics.mean_squared_error(
                        Y.iloc[val], reg.predict(X.iloc[val]))**0.5
                best_vars.append(reg.bestvar)        
            i=i+1
        cv_RMSE=np.mean(cv_RMSE_all)
        rpd=np.std(self.df[self.y_name])/cv_RMSE
        if verbose:
            print("RMSE: ",cv_RMSE)
            print("RPD: ",rpd)
            print("Selected freqs: ",best_vars)
            k=0
            for day in self.df[self.date_name].unique():    
                print("Date: {0}, Measurements: {1:.0f}, RMSE: {2:.2f}, selected vars: {3}"
                      .format(
                        np.datetime_as_string(day,unit='D'),
                        sum(self.df[self.date_name]==day),
                            cv_RMSE_all[k],
                            len(best_vars[k])))
                k=k+1
              
        return(rpd)

