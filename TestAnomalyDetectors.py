import pandas as pd
import os
from tqdm import tqdm
from anomaly_detector.AnomalyDetectorVanillaNF import AnomalyDetectorVanillaNF
from anomaly_detector.AnomalyDetectorVanillaNFnoNoise import AnomalyDetectorVanillaNFnoNoise
from anomaly_detector.AnomalyDetectorPSCAL import AnomalyDetectorPSCAL


def get_detectors():
    return [
        #AnomalyDetectorVanillaNF(),
        #AnomalyDetectorVanillaNFnoNoise(),
        AnomalyDetectorPSCAL()
    ]


def run_anomaly_detectors(dataset_id, n_sevs, n_runs, tag):
    detectors = get_detectors()
    output_file = f'res_files/{tag}/{dataset_id}.csv'

    for run_idx in tqdm(range(n_runs)):
        for detector in detectors:
            name = detector.get_name()
            for severity in range(n_sevs):
                detector.init(f"{dataset_id}_{severity}_{run_idx}")
                acc, auroc = detector.train_eval(run_idx, severity)

                print(f'{name} Acc: {acc * 100:.2f}% AUROC: {auroc * 100:.2f}%')

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