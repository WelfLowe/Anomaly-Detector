import numpy as np
from generator.Function import Function

class RandomNoise(Function):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu  # default mu 0
        self.sigma = sigma  # default sigma 1

    def eval(self, x):
        return np.random.normal(self.mu, self.sigma, 1)
    
    def generateN(self, N):
        return np.random.normal(self.mu, self.sigma, N)