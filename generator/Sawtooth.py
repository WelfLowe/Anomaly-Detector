import numpy as np
from generator.Function import Function

class Sawtooth(Function):
    def __init__(self, T=7.0 * 24, phase=0.0, c=1.0, iterations=1000):
        self.T = T  # period, default 7 days (in hours)
        self.phase = phase  # phase shift, default 0
        self.c = c  # constant factor, default 1
        self.iterations = iterations  # number of iterations
        
        # Create an array of n values from 1 to iterations
        n = np.arange(1, iterations + 1)
        
        # Compute the factors alternating between -1 and 1, divided by n
        z = (-1) ** (n + 1)  # Generates -1 for odd and 1 for even n
        self.factors = z / n

    def eval(self, x):
        # Convert x to a NumPy array if it's not already (to handle scalar or array inputs)
        x = np.asarray(x)
        
        # Create an array of n values from 1 to iterations
        n = np.arange(1, self.iterations + 1)
        
        # Compute the sine terms using broadcasting
        sine_terms = np.sin((x[..., np.newaxis] + self.phase) * n * 2 * np.pi / self.T)
        
        # Sum the product of factors and sine terms along the n-axis
        res = np.sum(self.factors * sine_terms, axis=-1)
        
        # Return the result scaled by the constant factor c and (2 / pi)
        return (-1)*self.c * (2 / np.pi) * res
