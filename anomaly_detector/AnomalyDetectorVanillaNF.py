from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.config as c
from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorVanillaNF(AnomalyDetector):
    def get_name(self) -> str:
        return "vanillaNF"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        def train_one_epoch(model, optimizer, dataloader):
            model.train()
            for data, _ in dataloader:
                optimizer.zero_grad()
                data = data.to(c.device)
                z = model(data)
                jac = model.nf.jacobian(run_forward = False)
                loss = 0.5 * torch.sum(z**2, dim=(1,)) - jac
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
    
        def evaluate(model, dataloader):
            model.eval()
            loss_list = []
            for data, _ in dataloader:
                data = data.to(c.device)
                z = model(data)
                jac = model.nf.jacobian(run_forward=False)
                loss = 0.5 * torch.sum(z**2, dim=(1,)) - jac
                loss_list += list(utl.t2np(loss))
            return loss_list

        def train_vanilla(data_holder, epochs, n_coupling_blocks, clamp_alpha, fc_internal, dropout, learning_rate, batch_size):
            n_features = data_holder.get_n_features()
            model = mdl.DifferNet(n_features, n_coupling_blocks, clamp_alpha, fc_internal, dropout)
            model.to(c.device)
            optimizer = torch.optim.Adam(
                model.nf.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-5,
            )

            for _ in range(epochs):
                dataloader = torch.utils.data.DataLoader(
                    utl.CustomDataset(data_holder.get_data()),
                    batch_size=batch_size,
                    shuffle=True,
                )
                train_one_epoch(model, optimizer, dataloader)
            return model

        # training
        data_norm = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        data_norm = data_norm.reshape(self.K, self.N)
        data_holder = utl.DataHolder(data_norm)

        model = train_vanilla(data_holder, c.epochs, c.n_coupling_blocks, c.clamp_alpha, c.fc_internal, c.dropout, c.learning_rate, c.batch_size)

        # validation
        val_data_norm = (self.val_data - self.data.min()) / (self.data.max() - self.data.min())
        val_data_norm = val_data_norm.reshape(val_data_norm.shape[0], self.N)
        data_holder = utl.DataHolder(val_data_norm)
        dataloader = torch.utils.data.DataLoader(
                    utl.CustomDataset(data_holder.get_data()),
                    batch_size=c.batch_size,
                    shuffle=False,
        )
        scores_val = evaluate(model, dataloader)
        scores_val = np.array(scores_val)
        
        anomalies =  scores_val > 3*np.std(scores_val)
        prediction_labels = np.where(anomalies, 0, 1)
        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(scores_val, self.val_labels)

        return accuracy, auroc