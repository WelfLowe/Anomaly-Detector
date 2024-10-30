from abc import ABC, abstractmethod
import numpy as np
from typing import List

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorIsoForest(AnomalyDetector):

    def get_name(self) -> str:
        return "Isolation Forest"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        np.random.shuffle(self.data)
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Assume 5% of the data is "bad"
        iso_forest.fit(self.data)
                
        # Predict: -1 for anomalies (bad time series), 1 for normal data (good time series)
        predictions = iso_forest.predict(self.val_data)

        # Convert predictions: -1 (anomaly) -> 1 (bad), 1 (normal) -> 0 (good)
        prediction_labels = np.where(predictions == -1, 1, 0)
        
        anomaly_scores = iso_forest.decision_function(self.val_data)
        
        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(anomaly_scores, self.val_labels)

        return accuracy, auroc