''' GAUSSIAN MIXTURE MODEL CLUSTERING '''
# Expected minimisation approach GMM for clustering

import numpy as np
from scipy.stats import multivariate_normal as mvn

class GMM():
    
    def __init__(self,C=3,max_iter=50,rtol=1e-3,n_restarts=10):
        self.C = C
        self.max_iter = max_iter
        self.rtol = rtol
        self.n_restarts = n_restarts
        
    @staticmethod
    def e_step(X, pi, mu, sigma):
        """
        Computes posterior probabilities from data and parameters.

        Args:
            X: observed data (N, D).
            pi: prior probabilities (C,).
            mu: mixture component means (C, D).
            sigma: mixture component covariances (C, D, D).

        Returns:
            Posterior probabilities (N, C).
        """
        
        N = X.shape[0]
        C = mu.shape[0]
        q = np.zeros((N, C))
        
        # Equation (6)
        for c in range(C):
            q[:, c] = mvn(mu[c], sigma[c]).pdf(X) * pi[c]        
        return q / np.sum(q, axis=-1, keepdims=True) 
    
    @staticmethod
    def m_step(X, q):
        """
        Computes parameters from data and posterior probabilities.

        Args:
            X: data (N, D).
            q: posterior probabilities (N, C).

        Returns:
            tuple of
            - prior probabilities (C,).
            - mixture component means (C, D).
            - mixture component covariances (C, D, D).
        """    
        
        N, D = X.shape
        C = q.shape[1]    
        sigma = np.zeros((C, D, D))
        
        # Equation (16)
        pi = np.sum(q, axis=0) / N
        
        # Equation (17)
        mu = q.T.dot(X) / np.sum(q.T, axis=1, keepdims=True)
        
        # Equation (18)
        for c in range(C):
            delta = (X - mu[c])
            sigma[c] = (q[:, [c]] * delta).T.dot(delta) / np.sum(q[:, c])
            
        return pi, mu, sigma    
    
    @staticmethod
    def lower_bound(X, pi, mu, sigma, q):
        """
        Computes lower bound from data, parameters and posterior probabilities.

        Args:
            X: observed data (N, D).
            pi: prior probabilities (C,).
            mu: mixture component means (C, D).
            sigma: mixture component covariances (C, D, D).
            q: posterior probabilities (N, C).

        Returns:
            Lower bound.
        """    
        
        N, C = q.shape
        ll = np.zeros((N, C))
        
        # Equation (19)
        for c in range(C):
            ll[:,c] = mvn(mu[c], sigma[c]).logpdf(X)
        return np.sum(q * (ll + np.log(pi) - np.log(np.maximum(q, 1e-8))))
    
    @staticmethod
    def random_init_params(X, C):
        D = X.shape[1]
        pi = np.ones(C) / C
        mu = mvn(mean=np.mean(X, axis=0), cov=[np.var(X[:, 0]), 
                                               np.var(X[:, 1])]).rvs(C).reshape(C, D)
        sigma = np.tile(np.eye(2), (C, 1, 1))
        return pi, mu, sigma
    
    def fit(self,X):
        
        self.proba = None  # posterior probabilities
        self.prior = None  # prior probabilities
        self.means = None  # mixture component means
        self.covariances = None # mixture component covariances
        lb_best = -np.inf
        
        for _ in range(self.n_restarts):
            pi, mu, sigma = self.random_init_params(X,self.C)
            prev_lb = None
            
            try:
                for _ in range(self.max_iter):
                    q = self.e_step(X, pi, mu, sigma)
                    pi, mu, sigma = self.m_step(X, q)
                    lb = self.lower_bound(X, pi, mu, sigma, q)
                    
                    if lb > lb_best:
                        self.proba = q
                        self.prior = pi
                        self.means = mu
                        self.covariances = sigma
                        lb_best = lb
                        
                    if prev_lb and np.abs((lb - prev_lb) / prev_lb) < self.rtol:
                        break
                    
                    prev_lb = lb
            except np.linalg.LinAlgError:
                # Singularity. One of the components collapsed
                # onto a specific data point. Start again ...
                pass