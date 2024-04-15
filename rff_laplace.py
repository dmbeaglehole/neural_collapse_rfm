from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy.stats import cauchy, laplace
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

import torch
import numpy as np

class RFF(BaseEstimator):
    def __init__(self, gamma = 1, D = 50, metric = "laplace"):
        self.gamma = gamma
        self.metric = metric
        #Dimensionality D (number of MonteCarlo samples)
        self.D = D
        self.fitted = False
        
    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        d = X.shape[1]
        #Generate D iid samples from p(w) 
        if self.metric == "rbf":
            self.w = np.sqrt(2*self.gamma)*np.random.normal(size=(self.D,d))
        elif self.metric == "laplace":
            self.w = torch.from_numpy(cauchy.rvs(scale = self.gamma, size=(self.D,d))).cuda().to(X.dtype)
        
        #Generate D iid samples from Uniform(0,2*pi) 
        self.u = torch.from_numpy(2*np.pi*np.random.rand(self.D)).cuda().to(X.dtype)
        self.fitted = True
        return self
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the feature map Z")
        #Compute feature map Z(x):
        Z = np.sqrt(2/self.D)*torch.cos((X@self.w.T + self.u.unsqueeze(0)))
        return Z