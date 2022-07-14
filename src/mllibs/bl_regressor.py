
''' Bayesian Linear Regression Model '''
# Two hyperparameter based linear regression model

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,RegressorMixin
from numpy.linalg import cholesky, det, lstsq, inv, eigvalsh, pinv
from scipy.optimize import minimize

class BR(BaseEstimator,RegressorMixin):

    # instantiation values
    def __init__(self,lamd=1.0e-5,alph=1e-5,maxiter=2000,rtol=1.0e-5,verbose=True):    
        self.maxiter = maxiter # class contains only tunable hyperparameters (max convergence iteration)
        self.rtol = rtol       # convergence tolerance for hyperparameters
        self.lamd = lamd   # hyperparameter 
        self.alph = alph     # hyperparameter
        self.verbose = verbose # can be activated to check coverged hyperparameters
    
    # compute mean cofficients/covariance matrix of posterior mean
    @staticmethod
    def posterior(X,y,lamd,alph):
        ndim = X.shape[1]
        S_N_inv = lamd * np.eye(ndim) + alph * X.T.dot(X) 
        S_N = inv(S_N_inv)                                      
        m_N = alph * S_N.dot(X.T).dot(y)                
        return m_N, S_N

    ''' train a bayesian ridge regression model + nearest classification '''
    
    def fit(self,X,y):

        ''' A. Check Input Data Compatibility '''
        if(type(X) is np.ndarray):
            self.X = X;self.y = y
        else:
            self.X = X.values; self.y = y.values
        ntot,ndim = self.X.shape

        # set initial value for hyperparameters
        eig0 = np.linalg.eigvalsh(self.X.T.dot(self.X))  # diagonal component (ndim,)

        # tune hyperparameters via convergence tolerance.
        for niter in range(self.maxiter):

            alph1 = self.alph
            lamd1 = self.lamd
            eig = eig0*self.alph

            # make prediction on training data
            self.m_N, self.S_N = self.posterior(self.X,self.y,self.lamd,self.alph)

            gamma = np.sum(eig/(eig+self.lamd))
            self.lamd = gamma / np.sum(self.m_N ** 2)
            Ibeta = 1.0 / (ntot-gamma) * np.sum((self.y - self.X.dot(self.m_N)) ** 2)
            self.beta = 1.0/Ibeta

            # define exit condition
            if np.isclose(lamd1,self.lamd,self.rtol) and np.isclose(alph1,self.alph,self.rtol):
                if(self.verbose is True):
                    print(f'{self.rtol} achieved at {niter+1} iterations.')
                    print(f'Converged Hyperparameters: {self.lamd,self.alph}')
                return self

        return self

    ''' make new predictions; mean + variance of posterior predictive distribution '''
    
    def predict(self,X):
        if(type(X) is np.ndarray):
            self.X = X
        else:
            self.X = X.values
        self.mu_s = X.dot(self.m_N)
        self.cov_s = 1.0 / self.alph + np.sum(X.dot(self.S_N) * X, axis=1)
        return self.mu_s