import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from anomaly_detector.AnomalyDetector import (
    AnomalyDetectorPSCAL,
    AnomalyDetectorVanillaNF,
    AnomalyDetectorVanillaNFnoNoise
)
from anomaly_detector.DatasetInfo import DatasetInfo

def get_detectors():
    return [
        AnomalyDetectorVanillaNF(),
        AnomalyDetectorVanillaNFnoNoise(),
        AnomalyDetectorPSCAL()
    ]

def load_dataset(base_path: str, dataset_id: str, severity: int, run_idx: int) -> DatasetInfo:
    suffix = f"{dataset_id}_{severity}_{run_idx}"
    data = np.load(os.path.join(base_path, f"train_{suffix}.npy"))
    
    return DatasetInfo(
        data=data,
        labels=np.load(os.path.join(base_path, f"train_{suffix}_labels.npy")),
        val_data=np.load(os.path.join(base_path, f"val_{suffix}.npy")),
        val_labels=np.load(os.path.join(base_path, f"val_{suffix}_labels.npy")),
        k=data.shape[0],
        n=data.shape[1],
        data_min=data.min(),
        data_max=data.max()
    )

def run_anomaly_detectors(dataset_id, n_sevs, n_runs, tag, testset_dir):
    detectors = get_detectors()
    output_file = f'res_files/{tag}/{dataset_id}.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for run_idx in tqdm(range(n_runs)):
        for severity in range(n_sevs):
            # NOTE: Disk I/O happens only once per run/severity combination
            dataset = load_dataset(testset_dir, dataset_id, severity, run_idx)
            
            for detector in detectors:
                name = detector.get_name()
                acc, auroc = detector.train_eval(dataset, run_idx, severity)

                row = {
                    'run': run_idx,
                    'train_set': dataset_id,
                    'severity': severity,
                    'alg': name,
                    'acc': acc,
                    'auroc': auroc
                }
                
                df = pd.DataFrame([row])
                is_new_file = not os.path.exists(output_file)
                df.to_csv(output_file, mode='a', index=False, header=is_new_file)