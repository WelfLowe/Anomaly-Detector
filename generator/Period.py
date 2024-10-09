import math
from generator.Function import Function

class Period(Function):
    def __init__(self, T=30.0 * 24, phase=0.0, c=1.0):
        self.T = T  # period, default 30 days (in hours)
        self.phase = phase  # phase shift, default 0
        self.c = c  # constant factor, default 1

    def eval(self, x):
        res = self.c * math.sin(2 * math.pi * (x + self.phase) / self.T)
        return res