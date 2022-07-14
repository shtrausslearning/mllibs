from sklearn.base import BaseEstimator,RegressorMixin
from numpy.linalg import cholesky, det, lstsq, inv, pinv
from scipy.optimize import minimize
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
pi = 4.0*np.arctan(1.0)
import warnings
warnings.filterwarnings("ignore")

# Universal Kriging Model (Polynomial Regression + Full Gaussian Process Regression Model)
# Commonly used Ensemble Approach for geospatial interpolation and 

class Kriging(BaseEstimator,RegressorMixin):
    
    def __init__(self,kernel='rbf',theta=10.0,sigma=10.0,sigma_n=1,opt=True,polyorder=2):
        self.theta = theta
        self.sigma = sigma
        self.sigma_n = sigma_n
        self.opt = opt
        self.polyorder = polyorder 
        Kriging.kernel = kernel 

    ''' local covariance functions '''
    @staticmethod
    def covfn(X0,X1,theta=1.0,sigma=1.0):

        ''' Radial Basis Covariance Function '''
        if(Kriging.kernel == 'rbf'):
            r = np.sum(X0**2,1).reshape(-1,1) + np.sum(X1**2,1) - 2 * np.dot(X0,X1.T)
            return sigma**2 * np.exp(-0.5/theta**2*r)

        ''' Matern Covariance Class of Funtions '''
        if(Kriging.kernel == 'matern'):
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
            
    ''' Train the Model'''
    def fit(self,X,y):
        
        ''' Working w/ numpy matrices'''
        if(type(X) is np.ndarray):
            self.X = X;self.y = y
        else:
            self.X = X.values; self.y = y.values
        self.ntot,ndim = self.X.shape
        
        # Collocation Matrix
        self.poly = PolynomialFeatures(self.polyorder)
        self.H = self.poly.fit_transform(self.X)
        
        ''' Optimisation Objective Function '''
        # Optimisation of hyperparameters via the objective funciton
        def llhobj(X,y,noise):
            
            # Simplified Variant
            def llh_dir(hypers):
                K = self.covfn(X,X,theta=hypers[0],sigma=hypers[1]) + noise**2 * np.eye(self.ntot)
                return 0.5 * np.log(det(K)) + \
                    0.5 * y.T.dot(inv(K).dot(y)).ravel()[0] + 0.5 * self.ntot * np.log(2*pi)

            # Full Likelihood Equation
            def nll_full(hypers):
                K = self.covfn(X,X,theta=hypers[0],sigma=hypers[1]) + noise**2 * np.eye(self.ntot)
                L = cholesky(K)
                return np.sum(np.log(np.diagonal(L))) + \
                    0.5 * y.T.dot(lstsq(L.T, lstsq(L,y)[0])[0]) + \
                    0.5 * self.ntot * np.log(2*pi)
            
            return llh_dir # return one of the two, simplified variant doesn't always work well
        
        ''' Update hyperparameters based on set objective function '''
        if(self.opt==True):
            # define the objective funciton
            objfn = llhobj(self.X,self.y,self.sigma_n)
            # search for the optimal hyperparameters based on given relation
            res = minimize(fun=objfn,x0=[1,1],
                           method='Nelder-Mead',tol=1e-6)
            self.theta,self.sigma = res.x # update the hyperparameters to 

        self.HT = self.H.T
        self.Kmat = self.covfn(self.X,self.X,self.theta,self.sigma) \
                  + self.sigma_n**2 * np.eye(self.ntot) # Covariance Matrix (Train/Train)
        self.IKmat = pinv(self.Kmat) # Pseudo Matrix Inversion (More Stable)

        self.HK = np.dot(self.HT,self.IKmat) # HK^-1
        HKH = np.dot(self.HK,self.H)     # HK^-1HT
        self.A = inv(HKH)             # Variance-Covariance Weighted LS Matrix

        self.W = np.dot(self.IKmat,self.y)
        Q = np.dot(self.HT,self.W)
        self.beta = np.dot(self.A,Q)               # Regression coefficients
        self.V = self.W - np.dot(self.IKmat,self.H).dot(self.beta) # K^{-1} (Y - H^{T} * beta)
        
        return self  # return class & use w/ predict()

    ''' Posterior Prediction;  '''
    # Make a prediction based on what the model has learned 
    def predict(self,Xm):
        
        ''' Working w/ numpy matrices'''
        if(type(Xm) is np.ndarray):
            self.Xm = Xm
        else:
            self.Xm = Xm.values
        self.mtot,ndim = self.Xm.shape
        
        self.Hm = self.poly.fit_transform(self.Xm) # Collocation Matrix
        self.Kmat = self.covfn(self.X,self.Xm,self.theta,self.sigma) # Covariance Matrix (Train/Test)
        yreg = np.dot(self.Hm,self.beta)               # Mean Prediction based on Regression
        ykr = np.dot(self.Kmat.T,self.V)              # posterior mean predictions for an explicit mean 

        return yreg + ykr