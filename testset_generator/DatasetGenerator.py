from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


class DatasetGenerator(ABC):
    @abstractmethod
    def getFunction(self, severeness: int):
        """Generates N correct data series."""
        pass

    def generateN(self, N: int, severeness: int):
        """Generates data series each with the given severeness s."""
        self.N = N
        f = self.getFunction(severeness)
        return f.generateN(N)
    
    def generateKN(self, K, N, fraction, severeness: int, verbose: bool, name : str = ""):
        """Generates K series of length N of data with fraction outliers and a certen severeness."""
        self.N = N
        self.K = K
        res = np.zeros([K,N])
        xs = np.arange(N)

        color = 'r'
        for i in tqdm(range(K)): 
            if i >= K*fraction:
                color = 'b'
                severeness = 0
            ys = self.generateN(N, severeness)
            if verbose:
                plt.plot(xs, ys, color, linewidth=0.2)
            res[i]=ys
        if verbose:
            plt.show()
            plt.close()
        if name != "":
            np.save(name+".npy", res)
            labels =  np.concatenate([np.ones(int(K*fraction)), np.zeros(int(K*(1-fraction)))])
            np.save(name+"_labels.npy",labels)
            
        return res   
    
    def load(self, name : str):
        res = np.load(name+".npy")
        labels = np.load(name+"_labels.npy")
        return [res, labels]
