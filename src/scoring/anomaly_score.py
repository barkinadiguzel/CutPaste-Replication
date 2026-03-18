import torch
import torch.nn as nn

class GaussianDensityEstimator:
    def __init__(self, mean, cov):
        self.mu = mean
        self.cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.size(0)))  

    def score(self, x):
        delta = x - self.mu
        score = torch.sum(delta @ self.cov_inv * delta, dim=1)  
        return score
