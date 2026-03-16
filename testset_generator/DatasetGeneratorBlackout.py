import numpy as np
from testset_generator.DatasetGenerator import DatasetGenerator

class DatasetGeneratorBlackout(DatasetGenerator):
    
    def getFunction(self, severeness: int):
        pass

    def generateKN(self, K, N, fraction, severeness: int, verbose=False, name=""):
        self.N = N
        self.K = K
        
        xs = np.arange(N).reshape(1, N)
        
        n_anomalies = int(K * fraction)
        sev_array = np.zeros((K, 1))
        sev_array[:n_anomalies] = severeness
        
        # NOTE: Base signal
        ys = 50 + np.random.normal(0, 1, (K, N))
        
        # NOTE: Monthly 
        p1 = (30 + np.random.normal(0, 2, (K, 1))) * 24
        a1 = 15 + np.random.normal(0, 2, (K, 1))
        ys += a1 * np.sin(2 * np.pi * xs / p1)
        
        # NOTE: Weekly
        p2 = (7 + np.random.normal(0, 1, (K, 1))) * 24
        a2 = 10 + np.random.normal(0, 1, (K, 1))
        ys += np.where((xs % p2) < (2 * 24), a2, 0)
        
        # NOTE: Daily
        p3 = (1 + np.random.normal(0, 0.1, (K, 1))) * 24
        a3 = 5 + np.random.normal(0, 0.1, (K, 1))
        ys += a3 * np.sin(2 * np.pi * (xs - (-6)) / p3)
        
        # NOTE: Blackout multiplier
        T = N / (sev_array + 1)
        tau = 10
        c = -0.1 * sev_array
        
        # TODO: Guard against division by zero if sev_array can theoretically be -1
        b = 1 + np.where((xs % T) < tau, c, 0)
        
        ys = (ys * b) + 150
        
        if name != "":
            np.save(name + ".npy", ys)
            labels = np.zeros(K)
            labels[:n_anomalies] = 1
            np.save(name + "_labels.npy", labels)
            
        return ys

    def load(self, name: str):
        data = np.load(name + ".npy")
        labels = np.load(name + "_labels.npy")
        return [data, labels]