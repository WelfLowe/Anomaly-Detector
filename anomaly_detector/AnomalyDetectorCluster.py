from abc import ABC, abstractmethod
import numpy as np
from typing import List

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorCluster(AnomalyDetector):
    def get_name(self) -> str:
        return "Cluster"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        # Standardize the data (important for KMeans)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)

        # Separate majority and minority classes
        good_data = data_scaled[self.labels == 0]
        bad_data = data_scaled[self.labels == 1]

        good_labels = self.labels[self.labels == 0]
        bad_labels = self.labels[self.labels == 1]

        # Oversample the minority class (bad_data)
        bad_data_oversampled, bad_labels_oversampled = resample(bad_data, bad_labels,
                                                                replace=True,  # Allow resampling with replacement
                                                                n_samples=len(good_data),  # Match the number of "good" data points
                                                                random_state=42)

        # Combine the oversampled data with the majority class data
        data_oversampled = np.vstack((good_data, bad_data_oversampled))
        labels_oversampled = np.hstack((good_labels, bad_labels_oversampled))

        # Shuffle the data to ensure random ordering
        indices = np.random.permutation(len(labels_oversampled))
        data_oversampled = data_oversampled[indices]
        labels_oversampled = labels_oversampled[indices]

        # Predict training cluster labels
        kmeans_oversampled = KMeans(n_clusters=2, random_state=42)
        kmeans_oversampled.fit(data_oversampled)

        val_data_scaled = scaler.transform(self.val_data)
        val_cluster_labels = kmeans_oversampled.predict(val_data_scaled)
        val_cluster_distances = kmeans_oversampled.transform(val_data_scaled)  

        accuracy = self.get_accuracy(val_cluster_labels, self.val_labels)
        auroc = self.get_auroc(val_cluster_distances[:,0], self.val_labels)

        return accuracy, auroc