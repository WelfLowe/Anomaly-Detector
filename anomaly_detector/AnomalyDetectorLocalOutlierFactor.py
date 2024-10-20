import numpy as np
from typing import List

from sklearn.utils import resample
from sklearn.neighbors import LocalOutlierFactor

from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorLocalOutlierFactor(AnomalyDetector):

    def get_name(self) -> str:
        return "Local Outlier Factor"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        np.random.shuffle(self.data)
        lof = LocalOutlierFactor(n_neighbors=50, novelty=True, contamination=0.05)
        lof.fit(self.data)

        # Predict: -1 for anomalies (bad time series), 1 for normal data (good time series)
        predictions = lof.predict(self.val_data)

        # Convert predictions: -1 (anomaly) -> 1 (bad), 1 (normal) -> 0 (good)
        prediction_labels = np.where(predictions == -1, 1, 0)        
        anomaly_scores = lof.decision_function(self.val_data)

        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(anomaly_scores, self.val_labels)

        return accuracy, auroc