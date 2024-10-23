from abc import ABC, abstractmethod
from typing import List

import numpy as np

import torch
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.config as c

from anomaly_detector.AnomalyDetector import AnomalyDetector

class AnomalyDetectorPSCAL(AnomalyDetector):
    def get_name(self) -> str:
        return "PSCAL"

    def train_eval(self) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        def train_one_epoch(model, optimizer, dataloader, std_cutoff):
            model.train()
            outliers = []

            last_ok_batch_mean = None
            last_ok_batch_std = None

            for data, index in dataloader:
                optimizer.zero_grad()

                data = data.to(c.device)

                z = model(data)
                jac = model.nf.jacobian(run_forward=False)
                loss = 0.5 * torch.sum(z**2, dim=(1,)) - jac

                batch_mean = torch.mean(loss)
                batch_std = torch.std(loss)

                # find indices of outliers
                new_outliers = torch.where(loss > batch_mean + std_cutoff * batch_std)[0]
                if len(new_outliers) > 0:
                    indices_of_outliers = list(
                        index[new_outliers.detach().cpu().numpy()].detach().cpu().numpy()
                    )
                    outliers += indices_of_outliers
                else:
                    loss = torch.mean(loss)
                    loss.backward()
                    optimizer.step()
                    last_ok_batch_mean = batch_mean
                    last_ok_batch_std = batch_std

            return outliers, last_ok_batch_mean, last_ok_batch_std

        def check_for_inliers(model, dataloader, mean, std, std_cutoff):
            model.eval()
            inliers = []
            for data, index in dataloader:
                data = data.to(c.device)
                z = model(data)
                jac = model.nf.jacobian(run_forward=False)
                loss = 0.5 * torch.sum(z**2, dim=(1,)) - jac

                # find indices of inliers
                new_inliers = torch.where(loss < mean + std * std_cutoff)[0]
                if len(new_inliers) > 0:
                    indices_of_inliers = list(
                        index[new_inliers.detach().cpu().numpy()].detach().cpu().numpy()
                    )
                    inliers += indices_of_inliers
            return inliers

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

        def train(data_holder, epochs, std_cutoff, n_coupling_blocks, clamp_alpha, fc_internal, dropout, learning_rate, batch_size):
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
                good_data_dataloader = torch.utils.data.DataLoader(
                    utl.CustomDataset(data_holder.get_good_data()),
                    batch_size=batch_size,
                    shuffle=True,
                )
                outliers, last_mean, last_std = train_one_epoch(
                    model, optimizer, good_data_dataloader, std_cutoff
                )
                #print("outliers ", len(outliers))
                data_holder.remove_outliers(outliers)
                if last_mean and data_holder.get_n_bad() > 0:
                    bad_data_dataloader = torch.utils.data.DataLoader(
                        utl.CustomDataset(data_holder.get_bad_data()),
                        batch_size=batch_size,
                        shuffle=True,
                    )
                    inliers = check_for_inliers(
                        model, bad_data_dataloader, last_mean, last_std, std_cutoff
                    )
                    #print("inliers ", len(inliers))
                    data_holder.add_inliers(inliers)

                    if data_holder.get_n_bad() == 0:
                        #print("No samples left in bad data")
                        continue
                    bad_data_dataloader = torch.utils.data.DataLoader(
                        utl.CustomDataset(data_holder.get_bad_data()),
                        batch_size=batch_size,
                        shuffle=True,
                    )
                    if data_holder.get_n_good() == 0:
                        #print("No samples left in good data")
                        continue
                    good_data_dataloader = torch.utils.data.DataLoader(
                        utl.CustomDataset(data_holder.get_good_data()),
                        batch_size=batch_size,
                        shuffle=True,
                    )
                    #bad_data_scores = evaluate(model, bad_data_dataloader)
                    #good_data_scores = evaluate(model, good_data_dataloader)
                    #print(np.mean(bad_data_scores))
                    #print(len(bad_data_scores))
                    #print(np.mean(good_data_scores))
                    #print(len(good_data_scores))
            return model

        # Normalize the data 
        data_norm = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        data_norm = data_norm.reshape(self.K, self.N)
        data_holder = utl.DataHolder(data_norm)
        
        model = train(data_holder, c.epochs, c.std_cutoff, c.n_coupling_blocks, c.clamp_alpha, c.fc_internal, c.dropout, c.learning_rate, c.batch_size)

        val_data_norm = (self.val_data - self.data.min()) / (self.data.max() - self.data.min())
        val_data_norm = val_data_norm.reshape(val_data_norm.shape[0], self.N)
        data_holder = utl.DataHolder(val_data_norm)
        dataloader = torch.utils.data.DataLoader(
                    utl.CustomDataset(data_holder.get_data()),
                    batch_size=c.batch_size,
                    shuffle=False,
        )
        scores_val = np.array(evaluate(model, dataloader))
        
        anomalies =  scores_val > 3*np.std(scores_val)
        prediction_labels = np.where(anomalies, 0, 1)
        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(scores_val, self.val_labels)

        return accuracy, auroc
