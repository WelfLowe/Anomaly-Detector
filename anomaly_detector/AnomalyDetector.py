import os
import csv
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from anomaly_detector.DatasetInfo import DatasetInfo
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.config as c


class AnomalyDetector(ABC):
    @abstractmethod
    def train_eval(self, dataset: DatasetInfo, r: int, s: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_params(self) -> Dict:
        params = {}
        for key in dir(c):
            if not key.startswith("__"):
                val = getattr(c, key)
                # NOTE: Cast complex objects like torch.device or modules to strings for JSON serialization
                if isinstance(val, (int, float, str, bool, list, tuple, dict)):
                    params[key] = val
                else:
                    params[key] = str(val)
        return params

    def normalize(self, data: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
        return (data - d_min) / (d_max - d_min)

    def get_accuracy(self, predicted_labels: np.ndarray, val_labels: np.ndarray) -> float:
        accuracy = np.mean(predicted_labels == val_labels)
        return float(max(accuracy, 1.0 - accuracy))

    def get_auroc(self, scores: np.ndarray, val_labels: np.ndarray) -> float:
        return float(max(
            roc_auc_score(val_labels, scores),
            roc_auc_score(val_labels, np.max(scores) - scores)
        ))

    def _evaluate_model(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> List[float]:
        model.eval()
        loss_list = []
        with torch.no_grad():
            for data, _ in dataloader:
                z = model(data.to(c.device))
                jac = model.nf.jacobian(run_forward=False)
                loss = 0.5 * torch.sum(z**2, dim=1) - jac
                loss_list.extend(utl.t2np(loss).tolist())
        return loss_list

    def _get_validation_scores(self, model: torch.nn.Module, dataset: DatasetInfo) -> Tuple[float, float]:
        val_data_norm = self.normalize(dataset.val_data, dataset.data_min, dataset.data_max).reshape(-1, dataset.n)
        data_holder = utl.DataHolder(val_data_norm)
        dataloader = torch.utils.data.DataLoader(
            utl.CustomDataset(data_holder.get_data()),
            batch_size=c.batch_size,
            shuffle=False,
        )
        
        scores_val = np.array(self._evaluate_model(model, dataloader))
        anomalies = scores_val > 3 * np.std(scores_val)
        prediction_labels = np.where(anomalies, 0, 1)
        
        return self.get_accuracy(prediction_labels, dataset.val_labels), self.get_auroc(scores_val, dataset.val_labels)


class AnomalyDetectorPSCAL(AnomalyDetector):
    def get_name(self) -> str:
        return "PSCAL"

    def train_eval(self, dataset: DatasetInfo, r: int, s: int) -> Tuple[float, float]:
        data_norm = self.normalize(dataset.data, dataset.data_min, dataset.data_max).reshape(dataset.k, dataset.n)
        data_holder = utl.DataHolder(data_norm)
        
        model = mdl.DifferNet(data_holder.get_n_features(), c.n_coupling_blocks, c.clamp_alpha, c.fc_internal, c.dropout).to(c.device)
        optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        for _ in range(c.epochs):
            loader_good = torch.utils.data.DataLoader(utl.CustomDataset(data_holder.get_good_data()), batch_size=c.batch_size, shuffle=True)
            outliers, last_mean, last_std = self._train_epoch(model, optimizer, loader_good, c.std_cutoff)
            data_holder.remove_outliers(outliers)

            if last_mean is not None and data_holder.get_n_bad() > 0:
                loader_bad = torch.utils.data.DataLoader(utl.CustomDataset(data_holder.get_bad_data()), batch_size=c.batch_size, shuffle=True)
                # TODO: Implement an annealing schedule for eps based on the current epoch
                recovered = self._check_inliers(model, loader_bad, last_mean, last_std, c.std_cutoff, c.explore_eps)
                data_holder.add_inliers(recovered)

        scores = self._get_validation_scores(model, dataset)

        # NOTE: Force PyTorch to release VRAM before the next iteration starts
        del model
        del optimizer
        del data_holder
        torch.cuda.empty_cache()

        return scores

    def _train_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader, xi: float):
        model.train()
        outliers = []
        last_mean, last_std = None, None

        for data, index in dataloader:
            optimizer.zero_grad()
            z = model(data.to(c.device))
            jac = model.nf.jacobian(run_forward=False)
            sample_losses = 0.5 * torch.sum(z**2, dim=1) - jac

            batch_mean = torch.mean(sample_losses)
            batch_std = torch.std(sample_losses)
            threshold = batch_mean + xi * batch_std

            is_outlier = sample_losses > threshold
            is_inlier = ~is_outlier

            if is_outlier.any():
                outliers.extend(index[is_outlier.cpu()].tolist())

            if is_inlier.any():
                inlier_loss = torch.mean(sample_losses[is_inlier])
                inlier_loss.backward()
                optimizer.step()
                last_mean, last_std = batch_mean.detach(), batch_std.detach()

        return outliers, last_mean, last_std

    @torch.no_grad()
    def _check_inliers(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, mean: torch.Tensor, std: torch.Tensor, xi: float, eps: float) -> List[int]:
        model.eval()
        recovered = []
        threshold = mean + std * xi

        for data, index in dataloader:
            data = data.to(c.device)
            force_return = torch.rand(data.size(0), device=c.device) < eps
            test_cands = ~force_return
            passed = torch.zeros_like(force_return)

            if test_cands.any():
                z = model(data[test_cands])
                jac = model.nf.jacobian(run_forward=False)
                losses = 0.5 * torch.sum(z**2, dim=1) - jac
                passed[test_cands] = losses <= threshold

            is_recovered = force_return | passed
            if is_recovered.any():
                recovered.extend(index[is_recovered.cpu()].tolist())

        return recovered


class BaseVanillaNF(AnomalyDetector):
    def train_eval_vanilla(self, dataset: DatasetInfo, r: int, s: int, filter_noise: bool) -> Tuple[float, float]:
        data_norm = self.normalize(dataset.data, dataset.data_min, dataset.data_max).reshape(dataset.k, dataset.n)
        if filter_noise:
            data_norm = data_norm[dataset.labels == 0]
            
        labels_to_use = dataset.labels if not filter_noise else dataset.labels[dataset.labels == 0]
        data_holder = utl.DataHolderLabeled(data_norm, labels_to_use)
        
        model = mdl.DifferNet(data_holder.get_n_features(), c.n_coupling_blocks, c.clamp_alpha, c.fc_internal, c.dropout).to(c.device)
        optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        log_file = f"logs/training_log_{'nonoise' if filter_noise else ''}{r}_{s}.csv"
        self._ensure_log_header(log_file)

        for epoch in range(c.epochs):
            loader = torch.utils.data.DataLoader(utl.CustomDatasetLabel(data_holder.get_data(), data_holder.get_labels()), batch_size=c.batch_size, shuffle=True)
            self._train_epoch(model, optimizer, loader, epoch, r, s, log_file)

        scores = self._get_validation_scores(model, dataset)

        del model
        del optimizer
        del data_holder
        torch.cuda.empty_cache()

        return scores

    def _ensure_log_header(self, log_file: str):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if not os.path.isfile(log_file):
            with open(log_file, mode="w", newline="") as f:
                csv.writer(f).writerow(["r", "s", "Epoch", "Batch", "Sample", "Label", "Loss"])

    def _train_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader, epoch: int, r: int, s: int, log_file: str):
        model.train()
        batch_logs = []

        for batch_idx, (data, labels, _) in enumerate(dataloader):
            optimizer.zero_grad()
            z = model(data.to(c.device))
            jac = model.nf.jacobian(run_forward=False)
            loss = 0.5 * torch.sum(z**2, dim=1) - jac
            
            torch.mean(loss).backward()
            optimizer.step()

            for i in range(data.size(0)):
                batch_logs.append([r, s, epoch, batch_idx, i, labels[i].item(), f"{loss[i].item():.6f}"])

        with open(log_file, mode="a", newline="") as f:
            csv.writer(f).writerows(batch_logs)


class AnomalyDetectorVanillaNF(BaseVanillaNF):
    def get_name(self) -> str:
        return "vanillaNF"
    
    def train_eval(self, dataset: DatasetInfo, r: int, s: int) -> Tuple[float, float]:
        return self.train_eval_vanilla(dataset, r, s, filter_noise=False)


class AnomalyDetectorVanillaNFnoNoise(BaseVanillaNF):
    def get_name(self) -> str:
        return "vanillaNFnoNoise"
    
    def train_eval(self, dataset: DatasetInfo, r: int, s: int) -> Tuple[float, float]:
        return self.train_eval_vanilla(dataset, r, s, filter_noise=True)