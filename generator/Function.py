from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):
    @abstractmethod
    def eval(self, x: float) -> float:
        """Evaluate the function at the given x value."""
        pass
    
    def generateN(self, N):
        res = np.zeros(N)
        for i in range(N):  # Loop from 0 to N-1
            res[i]=self.eval(i)
        return res