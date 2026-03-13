from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random
import torch
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.real_nvp_model as nvp_mdl
import anomaly_detector.NF.config as c

from anomaly_detector.AnomalyDetector import AnomalyDetector


class AnomalyDetectorPSCAL(AnomalyDetector):

    def get_name(self) -> str:
        return "PSCAL"

    def train_eval(self, r, s) -> List[float]:
        """Train and evaluate the anomaly detector."""
        """Return accuracy and AUROC."""

        def train_one_epoch(model, optimizer, dataloader, std_cutoff):
            model.train()
            outliers = []
            last_mean, last_std = None, None

            for data, index in dataloader:
                optimizer.zero_grad()
                data = data.to(c.device)

                z = model(data)
                jac = model.nf.jacobian(run_forward=False)
                sample_losses = 0.5 * torch.sum(z**2, dim=1) - jac

                batch_mean = torch.mean(sample_losses)
                batch_std = torch.std(sample_losses)
                threshold = batch_mean + std_cutoff * batch_std

                is_outlier = sample_losses > threshold
                is_inlier = ~is_outlier

                if is_outlier.any():
                    outliers.extend(index[is_outlier].tolist())

                if is_inlier.any():
                    inlier_loss = torch.mean(sample_losses[is_inlier])
                    inlier_loss.backward()
                    optimizer.step()
                    
                    last_mean = batch_mean.detach()
                    last_std = batch_std.detach()

            return outliers, last_mean, last_std

        @torch.no_grad()
        def check_for_inliers(model, dataloader, mean, std, std_cutoff, eps):
            model.eval()
            recovered_inliers = []
            threshold = mean + std * std_cutoff

            for data, index in dataloader:
                data = data.to(c.device)
                
                # NOTE: eps-greedy exploration
                rand_probs = torch.rand(data.size(0), device=c.device)
                force_return = rand_probs < eps
                test_candidates = ~force_return

                passed_test = torch.zeros_like(force_return)
                if test_candidates.any():
                    z = model(data[test_candidates])
                    jac = model.nf.jacobian(run_forward=False)
                    sample_losses = 0.5 * torch.sum(z**2, dim=1) - jac
                    passed_test[test_candidates] = sample_losses <= threshold

                is_recovered = force_return | passed_test
                if is_recovered.any():
                    recovered_inliers.extend(index[is_recovered].tolist())

            return recovered_inliers

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

        def train(data_holder, epochs, std_cutoff, n_coupling_blocks,
                  clamp_alpha, fc_internal, dropout, learning_rate,
                  batch_size):
            n_features = data_holder.get_n_features()
            model = mdl.DifferNet(n_features, n_coupling_blocks, clamp_alpha,
                                  fc_internal, dropout)
            #model = nvp_mdl.RealNVP(n_features, n_coupling_blocks, fc_internal)
            model.to(c.device)
            optimizer = torch.optim.Adam(
                model.nf.parameters(),
                #model.parameters(),
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

                data_holder.remove_outliers(outliers)

                if last_mean is not None and data_holder.get_n_bad() > 0:
                    bad_data_dataloader = torch.utils.data.DataLoader(
                        utl.CustomDataset(data_holder.get_bad_data()),
                        batch_size=batch_size,
                        shuffle=True, # NOTE: Shuffle isn't strictly necessary here anymore
                    )
                    
                    # TODO: Implement an annealing schedule for eps based on the current epoch
                    current_eps = 0.01 
                    
                    recovered = check_for_inliers(
                        model, bad_data_dataloader, last_mean, last_std, std_cutoff, current_eps
                    )
                    
                    data_holder.add_inliers(recovered)
            return model

        # Normalize the data
        data_norm = (self.data - self.data.min()) / (self.data.max() -
                                                     self.data.min())
        data_norm = data_norm.reshape(self.K, self.N)
        data_holder = utl.DataHolder(data_norm)

        model = train(data_holder, c.epochs, c.std_cutoff, c.n_coupling_blocks,
                      c.clamp_alpha, c.fc_internal, c.dropout, c.learning_rate,
                      c.batch_size)

        val_data_norm = (self.val_data - self.data.min()) / (self.data.max() -
                                                             self.data.min())
        val_data_norm = val_data_norm.reshape(val_data_norm.shape[0], self.N)
        data_holder = utl.DataHolder(val_data_norm)
        dataloader = torch.utils.data.DataLoader(
            utl.CustomDataset(data_holder.get_data()),
            batch_size=c.batch_size,
            shuffle=False,
        )
        scores_val = np.array(evaluate(model, dataloader))

        anomalies = scores_val > 3 * np.std(scores_val)
        prediction_labels = np.where(anomalies, 0, 1)
        accuracy = self.get_accuracy(prediction_labels, self.val_labels)
        auroc = self.get_auroc(scores_val, self.val_labels)

        return accuracy, auroc
