import numpy as np
from generator.Function import Function

class PulseTrain(Function):
    def __init__(self, T=7.0 * 24, tau=2.0 * 24, phase=-24.0, c=1.0, iterations=1000):
        self.T = T  # period, default 7 days (in hours)
        self.tau = tau  # pulse, default 2 days (in hours)
        self.phase = phase  # phase shift, default -tau / 2
        self.c = c  # constant factor, default 1
        self.iterations = iterations  # number of iterations, default 1000
        
        # Create an array of n values from 1 to iterations
        n = np.arange(1, iterations + 1)
        
        # Compute the factors vectorized
        self.factors = 2.0 * np.sin(np.pi * n * self.tau / self.T) / (np.pi * n)

    def eval(self, x):
        # Convert x to a NumPy array if it's not already (to handle scalar or array inputs)
        x = np.asarray(x)

        # Start with the first term: tau / T
        res = self.tau / self.T
        
        # Create an array of n values from 1 to iterations
        n = np.arange(1, self.iterations + 1)
        
        # Compute the cosine terms using broadcasting
        cos_terms = np.cos((x[..., np.newaxis] + self.phase) * 2 * np.pi * n / self.T)
        
        # Sum the product of factors and cosine terms along the n-axis
        res += np.sum(self.factors * cos_terms, axis=-1)

        # Return the result scaled by the constant factor c
        return self.c * res