from abc import ABC, abstractmethod
import numpy as np
from typing import List

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorOneClassSVM(AnomalyDetector):

    def get_name(self) -> str:
        return "One Class SVM"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        np.random.shuffle(self.data)
        svm = OneClassSVM(nu=0.05, kernel="rbf")  # nu is the proportion of outliers
        svm.fit(self.data)

        # Predict: -1 for anomalies (bad time series), 1 for normal data (good time series)
        predictions = svm.predict(self.val_data)

        # Convert predictions: -1 (anomaly) -> 1 (bad), 1 (normal) -> 0 (good)
        prediction_labels = np.where(predictions == -1, 1, 0)        
        anomaly_scores = svm.decision_function(self.val_data)

        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(anomaly_scores, self.val_labels)

        return accuracy, auroc