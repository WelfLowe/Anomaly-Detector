from abc import ABC, abstractmethod
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score


class AnomalyDetector(ABC):
    @abstractmethod
    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the detector's name."""
        pass

    def init(self, dataset_id):
        self.data = np.load("testsets/train_"+dataset_id+".npy")
        self.labels = np.load("testsets/train_"+dataset_id+"_labels.npy")
        self.K = self.data.shape[0]
        self.N = self.data.shape[1]
        self.val_data = np.load("testsets/val_"+dataset_id+".npy")
        self.val_labels = np.load("testsets/val_"+dataset_id+"_labels.npy")

    def get_accuracy(self, predicted_labels, val_labels):
        accuracy = np.mean(predicted_labels == val_labels)
        return max(accuracy,1.-accuracy)

    def get_auroc(self, scores, val_labels):
        return max(roc_auc_score(val_labels, scores),roc_auc_score(val_labels, np.max(scores)- scores)) 
    
