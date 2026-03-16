import numpy as np
from testset_generator.DatasetGenerator import DatasetGenerator

class DatasetGeneratorLift(DatasetGenerator):
    
    def getFunction(self, severeness: int):
        # NOTE: Unused in vectorized approach, kept for interface compliance
        pass

    def generateKN(self, K, N, fraction, severeness: int, verbose=False, name=""):
        self.N = N
        self.K = K
        
        xs = np.arange(N).reshape(1, N)
        
        n_anomalies = int(K * fraction)
        sev_array = np.zeros((K, 1))
        sev_array[:n_anomalies] = severeness
        
        # NOTE: Base signal
        offset = 200 + 2 * sev_array
        noise = np.random.normal(0, 1, (K, N))
        ys = offset + noise
        
        # NOTE: Monthly 
        p1 = (30 + np.random.normal(0, 0.3, (K, 1))) * 24
        a1 = 15 + np.random.normal(0, 0.3, (K, 1))
        ys += a1 * np.sin(2 * np.pi * xs / p1)
        
        # NOTE: Weekly
        p2 = (7 + np.random.normal(0, 0.2, (K, 1))) * 24
        a2 = 10 + np.random.normal(0, 0.2, (K, 1))
        pulse_width = 2 * 24
        modulo_time = xs % p2
        ys += np.where(modulo_time < pulse_width, a2, 0)
        
        # NOTE: Daily
        p3 = (1 + np.random.normal(0, 0.1, (K, 1))) * 24
        a3 = 5 + np.random.normal(0, 0.1, (K, 1))
        ys += a3 * np.sin(2 * np.pi * (xs - (-6)) / p3)
        
        if name:
            # TODO: Update main script to handle .npz reading instead of two .npy files
            labels = np.zeros(K)
            labels[:n_anomalies] = 1
            np.savez_compressed(name + ".npz", data=ys, labels=labels)
            
        return ys

    def load(self, name: str):
        # NOTE: Matching the new .npz save format
        data = np.load(name + ".npz")
        return [data['data'], data['labels']]