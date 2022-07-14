''' Gaussian Process Regression Class '''
# Standard Gaussian process regressor, automated hyperparameter
# optimisation (using scipy) or self defined three hyperparameter model

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,RegressorMixin
from numpy.linalg import cholesky, det, lstsq, inv, eigvalsh, pinv
from scipy.optimize import minimize
pi = 4.0*np.arctan(1.0)

# Usage similar to any sklearn model
class GPR(BaseEstimator,RegressorMixin):

    ''' Class Instantiation Related Variables '''
    def __init__(self,kernel='rbf',theta=10.0,sigma=10.0,sigma_n=1,opt=True):
        self.theta = theta            # Hyperparameter associated with covariance function
        self.sigma = sigma            #                       ''
        self.sigma_n = sigma_n        # Hyperparameter associated with cov.mat's diagonal component
        self.opt = opt                # Update hyperparameters with objective function optimisation
        GPR.kernel = kernel           # Selection of Covariance Function, class specific instantiation

    ''' local covariance functions '''
    @staticmethod
    def covfn(X0,X1,theta=1.0,sigma=1.0):

        ''' Radial Basis Covariance Function '''
        if(GPR.kernel == 'rbf'):
            r = np.sum(X0**2,1).reshape(-1,1) + np.sum(X1**2,1) - 2 * np.dot(X0,X1.T)
            return sigma**2 * np.exp(-0.5/theta**2*r)

        ''' Matern Covariance Class of Funtions '''
        if(GPR.kernel == 'matern'):
            lid=1
            r = np.sum(X0**2,1)[:,None] + np.sum(X1**2,1) - 2 * np.dot(X0,X1.T)
            if(lid==1):
                return sigma**2 * np.exp(-r/theta)
            elif(lid==2):
                ratio = r/theta
                v1 = (1.0+np.sqrt(3)*ratio)
                v2 = np.exp(-np.sqrt(3)*ratio)
                return sigma**2*v1*v2
            elif(lid==3):
                ratio = r/theta
                v1 = (1.0+np.sqrt(5)*ratio+(5.0/3.0)*ratio**2)
                v2 = np.exp(-np.sqrt(5)*ratio)
                return sigma**2*v1*v2
        else:
            print('Covariance Function not defined')
    
    ''' Train the GPR Model'''
    def fit(self,X,y):
        
        ''' Working w/ numpy matrices'''
        if(type(X) is np.ndarray):
            self.X = X;self.y = y
        else:
            self.X = X.values; self.y = y.values
        self.ntot,ndim = self.X.shape

        ''' Optimisation Objective Function '''
        def llhobj(X,y,noise):
           # Simplified Variant
            def llh_dir(hypers):
                K = self.covfn(X,X,theta=hypers[0],sigma=hypers[1]) + noise**2 * np.eye(self.ntot)
                return 0.5 * np.log(det(K)) + \
                    0.5 * y.T.dot(pinv(K).dot(y)).ravel()[0] + 0.5 * len(X) * np.log(2*pi)

            # Full Likelihood Equation
            def nll_full(hypers):
                K = self.covfn(X,X,theta=hypers[0],sigma=hypers[1]) + noise**2 * np.eye(self.ntot)
                L = cholesky(K)
                return np.sum(np.log(np.diagonal(L))) + \
                    0.5 * y.T.dot(lstsq(L.T, lstsq(L,y)[0],rcond=None)[0]) + \
                    0.5 * len(X) * np.log(2*pi)

            return nll_full # return one of the two, simplified variant doesn't always work well

        ''' Update hyperparameters based on set objective function '''
        if(self.opt==True):
            # define the objective funciton
            objfn = llhobj(self.X,self.y,self.sigma_n)
            # search for the optimal hyperparameters based on given relation
            res = minimize(objfn,[1,1],bounds=((1e-5,None),(1e-5, None)),method='L-BFGS-B')
            self.theta,self.sigma = res.x # update the hyperparameters to 

        ''' Get Training Covariance Matrix, K^-1 '''
        Kmat = self.covfn(self.X,self.X,self.theta,self.sigma) \
                 + self.sigma_n**2 * np.eye(self.ntot) # Covariance Matrix (Train/Train)
        self.IKmat = pinv(Kmat) # Pseudo Matrix Inversion (More Stable)
        return self  # return class & use w/ predict()

    ''' Posterior Prediction;  '''
    # Make a prediction based on what the model has learned 
    def predict(self,Xm):
        
        if(type(Xm) is np.ndarray):
            Xm = Xm
        else:
            Xm = Xm.values
        # Covariance Matrices x2 required; (Train/Train,Train/Test)
        mtot = Xm.shape[0]  # Number of Test Matrix Instances 
        K_s = self.covfn(self.X,Xm,self.theta,self.sigma)  # Covariance Matrix (Train/Test) 
        self.mu_s = K_s.T.dot(self.IKmat).dot(self.y)      # Posterior Mean Prediction
        return self.mu_s # return posterior mean