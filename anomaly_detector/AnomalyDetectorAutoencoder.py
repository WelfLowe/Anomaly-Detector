from abc import ABC, abstractmethod
from typing import List

import numpy as np

import keras
from keras import Sequential
from keras.src.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, BatchNormalization
from keras import optimizers

from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorAutoencoder(AnomalyDetector):
    def get_name(self) -> str:
        return "Autoencoder"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        # Normalize the data 
        data_norm = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        data_norm = data_norm.reshape(self.K, self.N, 1)

        # Define the Autoencoder model (LSTM-based)
        model = Sequential()
        model.add(Input(shape=(self.N, 1))) 
#        model.add(LSTM(64, activation='relu', input_shape=(self.N, 1), return_sequences=False))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(BatchNormalization())
        model.add(RepeatVector(self.N))  # Repeat vector to match the input shape for decoding
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))  # Output layer with the same time steps

        # Compile the model
        # optimizers.Adam
        model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5, clipnorm=1.0), loss='mse')

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model on the majority class (you can identify majority class data via labels or all data assuming unsupervised setting)
        model.fit(data_norm, data_norm, 
                    epochs=50, batch_size=128, validation_split=0.1, 
                    callbacks=[early_stopping], 
                    verbose=0)
        
        reconstructed_data = model.predict(data_norm, verbose=0)
        reconstruction_error = np.mean(np.power(data_norm - reconstructed_data, 2), axis=1)

        # Set a threshold for anomaly detection based on the reconstruction error
        threshold = np.percentile(reconstruction_error, 95)  # Example: top 5% highest errors are anomalies

        val_data_norm = (self.val_data - self.data.min()) / (self.data.max() - self.data.min())
        val_data_norm = val_data_norm.reshape(val_data_norm.shape[0], self.N, 1)

        reconstructed_data = model.predict(val_data_norm, verbose=0)
        reconstruction_error = np.mean(np.power(val_data_norm - reconstructed_data, 2), axis=1)

        # Classify time series as anomalies if their reconstruction error exceeds the threshold
        # anomalies is a boolean array, where True indicates a likely "bad" time series
        anomalies = reconstruction_error > threshold    
        prediction_labels = np.where(anomalies, 0, 1)

        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(reconstruction_error, self.val_labels)

        return accuracy, auroc
