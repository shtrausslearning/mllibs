''' Gaussian Process Binary Classifier '''
# Gaussian Process based Classifier utilising 
# rbf & matern based two covariance functions

import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.optimize import minimize
from scipy.stats import bernoulli
from scipy.special import expit as sigmoid
from numpy.linalg import inv,slogdet

class GPC(BaseEstimator,ClassifierMixin):
    
    def __init__(self,hyper=None,opt=True,verbose=False,sima_n=1.0e-4,return_std=False,kernel='rbf'):
        self.hyper = hyper        # list of hyperparameters [theta,sigma]
        self.opt = opt            # optimisation of hyperparameters tf
        self.verbose = verbose    # 
        self.sima_n = sima_n              # noise hyperparameter
        GPC.kernel = kernel 
        self.return_std = return_std  
        self.call_id = 0
        if(hyper is None and opt is False):
            assert(False)
        
    # Compute RBF Kernel 
    @staticmethod
    def covfn(X0,X1,hyper):    
        ''' Radial Basis Covariance Function '''
        if(GPC.kernel == 'rbf'):
            r = np.sum(X0**2,1)[:,None] + np.sum(X1**2,1) - 2 * np.dot(X0,X1.T)
            return hyper[1]**2 * np.exp(-0.5/hyper[0]**2*r)
        
        ''' Matern Covariance Class of Funtions '''
        if(GPC.kernel == 'matern'):
            lid=3
            r = np.sum(X0**2,1)[:,None] + np.sum(X1**2,1) - 2 * np.dot(X0,X1.T)
            if(lid==1):
                return hyper[1]**2 * np.exp(-r/hyper[0])
            elif(lid==2):
                ratio = r/hyper[0]
                v1 = (1.0+np.sqrt(3)*ratio)
                v2 = np.exp(-np.sqrt(3)*ratio)
                return hyper[1]**2*v1*v2
            elif(lid==3):
                ratio = r/hyper[0]
                v1 = (1.0+np.sqrt(5)*ratio+(5.0/3.0)*ratio**2)
                v2 = np.exp(-np.sqrt(5)*ratio)
                return hyper[1]**2*v1*v2
        else:
            print('Covariance Function not defined')
    
    # Compute the mode of posterior p(a|t)
    @staticmethod
    def posterior_mode(X,y,K_a, max_iter=10, tol=1e-9):

        a_h = np.zeros_like(y)
        I = np.eye(X.shape[0])

        for i in range(max_iter):
            
            W = np.diag((sigmoid(a_h) * (1 - sigmoid(a_h))).ravel())
            Q_inv = inv(I + W @ K_a)
            a_h_new = (K_a @ Q_inv).dot(y - sigmoid(a_h) + W.dot(a_h))
            a_h_diff = np.abs(a_h_new - a_h)
            a_h = a_h_new

            if not np.any(a_h_diff > tol):
                break

        return a_h
        
    # Train model 
    def fit(self,X,y):
        
        self.X_train = X
        self.y_train = y
        self.class_labels = np.unique(y)
        if(self.class_labels.shape[0]>2):
            print('Multi-Class Classification not implemented')
            
        # optimisation option, else use input hyperparameters
        if(self.opt):
            
            # objective function -Ve log(likelihood)
            def nll_fn(X,y):
 
                y = y.ravel()
                def nll(hyper):
                
                    K_a = self.covfn(X,X,hyper) + self.sima_n * np.eye(X.shape[0])
                    K_a_inv = inv(K_a)

                    # posterior mode depends on hyper (via K)
                    a_h = self.posterior_mode(X,y,K_a).ravel()
                    r = sigmoid(a_h) * (1 - sigmoid(a_h))
                    W = np.diag(r.ravel())
                    ll = - 0.5 * a_h.T.dot(K_a_inv).dot(a_h) \
                         - 0.5 * slogdet(K_a)[1] \
                         - 0.5 * slogdet(W + K_a_inv)[1] \
                         + y.dot(a_h) - np.sum(np.log(1.0 + np.exp(a_h)))

                    return -ll

                return nll

            # definite minimisation problem 
            res = minimize( nll_fn(X,y),[1,1], method='L-BFGS-B',
                            bounds=((1e-3, None),(1e-3, None)) )
            self.hyper = res.x  # update hyperparameter values
            self.nll = res.fun  # -Ve log(likelihood) value
        
        return self
    
    # Compute probability of class = 1 at X(input) [0,1]
    def predict_proba(self,X):
        self.call_id = 1        # predict_proba called
        a_mu, a_var = self.predict(X)
        kappa = 1.0 / np.sqrt(1.0 + np.pi*a_var/ 8)
        prob_1 = sigmoid(kappa * a_mu)
        prob_0 = (1.0 - prob_1[:,0])[:,None]
        prob_all = np.concatenate((prob_0,prob_1),axis=1)
        return prob_all
    
    # Mean/Variance of logits at X (-Val,Val)
    def predict(self,X):
        
        if(self.call_id is 0):
            lcall = 1
        elif(self.call_id is 1):
            lcall = 0
            
        K_a = self.covfn(self.X_train,
                         self.X_train,
                         self.hyper) + self.sima_n * np.eye(self.X_train.shape[0])
        K_s = self.covfn(self.X_train,X,self.hyper)
        a_h = self.posterior_mode(self.X_train,self.y_train,K_a)

        r = sigmoid(a_h) * (1 - sigmoid(a_h))        
        W_inv = inv(np.diag(r.ravel()))
        R_inv = inv(W_inv + K_a)

        a_test_mu = K_s.T.dot(self.y_train - sigmoid(a_h))
        a_test_var = (self.hyper[1] ** 2 + self.sima_n ) - np.sum((R_inv @ K_s) * K_s, axis=0)[:,None]
        a_test_mu_out = np.where(a_test_mu<0,0,1)
        
        # output depending on call orders
        if(self.return_std):
            if(lcall is 0):
                return a_test_mu, a_test_var 
                self.call_id = 0
            else:
                return a_test_mu_out, a_test_var # output [0,1]
        if(self.return_std is False): # used in CV only 
            if(lcall is 0):
                return a_test_mu
            else:
                return a_test_mu_out
                self.call_id = 0