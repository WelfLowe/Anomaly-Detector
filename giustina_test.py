import pandas as pd
from tqdm import tqdm
#from anomaly_detector.AnomalyDetectorAutoencoder import AnomalyDetectorAutoencoder
from anomaly_detector.AnomalyDetectorCluster import AnomalyDetectorCluster
from anomaly_detector.AnomalyDetectorClusterIsoForest import AnomalyDetectorIsoForest
from anomaly_detector.AnomalyDetectorLocalOutlierFactor import AnomalyDetectorLocalOutlierFactor
from anomaly_detector.AnomalyDetectorOneClassSVM import AnomalyDetectorOneClassSVM
from anomaly_detector.AnomalyDetectorVanillaNF import AnomalyDetectorVanillaNF
from anomaly_detector.AnomalyDetectorVanillaNFnoNoise import AnomalyDetectorVanillaNFnoNoise
from anomaly_detector.AnomalyDetectorPSCAL import AnomalyDetectorPSCAL

ad = []
#ad.append(AnomalyDetectorCluster())
#ad.append(AnomalyDetectorIsoForest())
#ad.append(AnomalyDetectorOneClassSVM())
#ad.append(AnomalyDetectorLocalOutlierFactor())
#ad.append(AnomalyDetectorAutoencoder())
#ad.append(AnomalyDetectorVanillaNF())
#ad.append(AnomalyDetectorVanillaNFnoNoise())
ad.append(AnomalyDetectorPSCAL())


def TestAnomalyDetector(n_runs):
    d.init(str(i) + "_" + str(s) + "_" + str(r))
    accuracy, auroc = d.train_eval(r, s)  #d.train_eval(r, s)
    row = [r, i, s, name, accuracy, auroc]
    l.append(row)
    print(f'{name} Accuracy: {accuracy * 100:.2f}% AUROC: {auroc * 100:.2f}%')

    df = pd.DataFrame(
        [row], columns=['run', 'train_set', 'severity', 'alg', 'acc', 'auroc'])
    df.to_csv(f'res_files_new/{i}.csv',
              mode='a',
              index=False,
              header=not pd.io.common.file_exists(f'res_files_new/{i}.csv'))
    l = []
