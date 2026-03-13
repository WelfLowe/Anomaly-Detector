from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.config as c
from anomaly_detector.AnomalyDetector import AnomalyDetector
import csv
import os


class AnomalyDetectorVanillaNFnoNoise(AnomalyDetector):

    def get_name(self) -> str:
        return "vanillaNFnoNoise"

    def train_eval(self, r, s) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        # def train_one_epoch(model, optimizer, dataloader, epoch):
        #     model.train()
        #     for data, _, labels in dataloader:
        #         optimizer.zero_grad()
        #         data = data.to(c.device)
        #         z = model(data)
        #         jac = model.nf.jacobian(run_forward=False)
        #         loss = 0.5 * torch.sum(z**2, dim=(1, )) - jac
        #         loss = torch.mean(loss)
        #         loss.backward()
        #         optimizer.step()

        def train_one_epoch(model,
                            optimizer,
                            dataloader,
                            epoch,
                            log_file=f"logs/training_log_nonoise{r}_{s}.csv"):
            model.train()

            with open(log_file, "a") as file:  # Open log file in append mode
                for batch_idx, (data, labels, _) in enumerate(dataloader):
                    optimizer.zero_grad()
                    data = data.to(c.device)

                    # Forward pass
                    z = model(data)
                    jac = model.nf.jacobian(run_forward=False)
                    loss = 0.5 * torch.sum(
                        z**2, dim=(1, )) - jac  # Loss for each sample
                    mean_loss = torch.mean(loss)  # Mean loss for batch

                    # Backward pass and optimization step
                    mean_loss.backward()
                    optimizer.step()

                    file_exists = os.path.isfile(
                        log_file)  # Check if file exists

                    # Open CSV file in append mode
                    with open(log_file, mode="a", newline="") as file:
                        writer = csv.writer(file)

                        # Write header only if the file is new
                        if not file_exists:
                            writer.writerow([
                                "r", "s", "Epoch", "Batch", "Sample", "Label",
                                "Loss"
                            ])

                        # Log each sample's details
                        for sample_idx in range(data.size(0)):
                            writer.writerow([
                                r, s, epoch, batch_idx, sample_idx,
                                labels[sample_idx].item(),
                                f"{loss[sample_idx].item():.6f}"
                            ])

        def evaluate(model, dataloader):
            model.eval()
            loss_list = []
            for data, _ in dataloader:
                data = data.to(c.device)
                z = model(data)
                jac = model.nf.jacobian(run_forward=False)
                loss = 0.5 * torch.sum(z**2, dim=(1, )) - jac
                loss_list += list(utl.t2np(loss))
            return loss_list

        def train_vanilla(data_holder, epochs, n_coupling_blocks, clamp_alpha,
                          fc_internal, dropout, learning_rate, batch_size):
            n_features = data_holder.get_n_features()
            model = mdl.DifferNet(n_features, n_coupling_blocks, clamp_alpha,
                                  fc_internal, dropout)
            model.to(c.device)
            optimizer = torch.optim.Adam(
                model.nf.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-5,
            )

            for epoch in range(epochs):
                dataloader = torch.utils.data.DataLoader(
                    utl.CustomDatasetLabel(data_holder.get_data(),
                                           data_holder.get_labels()),
                    batch_size=batch_size,
                    shuffle=True,
                )

                train_one_epoch(model, optimizer, dataloader, epoch)
            return model

        # training
        data_norm = (self.data - self.data.min()) / (self.data.max() -
                                                     self.data.min())
        data_norm = data_norm.reshape(self.K, self.N)
        data_norm = data_norm[self.labels == 0]
        data_holder = utl.DataHolderLabeled(data_norm, self.labels)

        model = train_vanilla(data_holder, c.epochs, c.n_coupling_blocks,
                              c.clamp_alpha, c.fc_internal, c.dropout,
                              c.learning_rate, c.batch_size)

        # validation
        val_data_norm = (self.val_data - self.data.min()) / (self.data.max() -
                                                             self.data.min())
        val_data_norm = val_data_norm.reshape(val_data_norm.shape[0], self.N)
        data_holder = utl.DataHolderLabeled(val_data_norm, self.val_labels)
        dataloader = torch.utils.data.DataLoader(
            utl.CustomDataset(data_holder.get_data()),
            batch_size=c.batch_size,
            shuffle=False,
        )
        scores_val = evaluate(model, dataloader)
        scores_val = np.array(scores_val)

        anomalies = scores_val > 3 * np.std(scores_val)
        prediction_labels = np.where(anomalies, 0, 1)
        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(scores_val, self.val_labels)

        return accuracy, auroc


if __name__ == "__main__":

    AD = AnomalyDetectorVanillaNFnoNoise()
    accuracy, auroc = AD.train_eval()
