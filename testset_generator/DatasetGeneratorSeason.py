import numpy as np
from testset_generator.DatasetGenerator import DatasetGenerator

class VectorizedSeasonalGenerator(DatasetGenerator):
    def _get_params(self, K, sev_array):
        raise NotImplementedError
        
    def getFunction(self, severeness: int):
        pass

    def generateKN(self, K, N, fraction, severeness: int, verbose=False, name=""):
        self.N = N
        self.K = K
        
        xs = np.arange(N).reshape(1, N)
        n_anomalies = int(K * fraction)
        sev_array = np.zeros((K, 1))
        sev_array[:n_anomalies] = severeness
        
        offset, noise_scale, p1, a1, p2, a2, p3, a3 = self._get_params(K, sev_array)
        
        ys = offset + np.random.normal(0, noise_scale, (K, N))
        
        ys += a1 * np.sin(2 * np.pi * xs / (p1 * 24))
        ys += np.where(((xs - 24) % (p2 * 24)) < (2 * 24), a2, 0)
        ys += a3 * np.sin(2 * np.pi * (xs - (-6)) / (p3 * 24))
        
        if name:
            labels = np.zeros(K)
            labels[:n_anomalies] = 1
            np.savez_compressed(name + ".npz", data=ys, labels=labels)
            
        return ys

    def load(self, name: str):
        data = np.load(name + ".npz")
        return [data['data'], data['labels']]


class DatasetGeneratorNoise(VectorizedSeasonalGenerator):
    def _get_params(self, K, sev_array):
        offset = 200
        noise_scale = 1 + sev_array
        
        p1 = 30 + np.random.normal(0, 2, (K, 1))
        a1 = 15 + np.random.normal(0, 2, (K, 1))
        
        p2 = 7 + np.random.normal(0, 1, (K, 1))
        a2 = 10 + np.random.normal(0, 1, (K, 1))
        
        p3 = 1 + np.random.normal(0, 0.1, (K, 1))
        a3 = 5 + np.random.normal(0, 0.1, (K, 1))
        
        return offset, noise_scale, p1, a1, p2, a2, p3, a3


class DatasetGeneratorAmplitude(VectorizedSeasonalGenerator):
    def _get_params(self, K, sev_array):
        offset = 200
        noise_scale = 1
        
        p1 = 30 + np.random.normal(0, 2, (K, 1))
        a1 = 15 + 2 * sev_array + np.random.normal(0, 2, (K, 1))
        
        p2 = 7 + np.random.normal(0, 1, (K, 1))
        a2 = 10 + sev_array + np.random.normal(0, 1, (K, 1))
        
        p3 = 1 + np.random.normal(0, 0.1, (K, 1))
        a3 = 5 + 0.1 * sev_array + np.random.normal(0, 0.1, (K, 1))
        
        return offset, noise_scale, p1, a1, p2, a2, p3, a3


class DatasetGeneratorShift(VectorizedSeasonalGenerator):
    def _get_params(self, K, sev_array):
        offset = 200
        noise_scale = 1
        
        # NOTE: The mathematical operations here modify the period length (frequency), 
        # not the phase shift as the class docstring originally implied.
        p1 = 30 + 2 * sev_array + np.random.normal(0, 2, (K, 1))
        a1 = 15 + np.random.normal(0, 2, (K, 1))
        
        p2 = 7 + sev_array + np.random.normal(0, 1, (K, 1))
        a2 = 10 + np.random.normal(0, 1, (K, 1))
        
        p3 = 1 + 0.1 * sev_array + np.random.normal(0, 0.1, (K, 1))
        a3 = 5 + np.random.normal(0, 0.1, (K, 1))
        
        return offset, noise_scale, p1, a1, p2, a2, p3, a3