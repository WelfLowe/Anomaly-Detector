from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from anomaly_detector.AnomalyDetectorAutoencoder import AnomalyDetectorAutoencoder
from anomaly_detector.AnomalyDetectorCluster import AnomalyDetectorCluster
from anomaly_detector.AnomalyDetectorClusterIsoForest import AnomalyDetectorIsoForest
from anomaly_detector.AnomalyDetectorLocalOutlierFactor import AnomalyDetectorLocalOutlierFactor
from anomaly_detector.AnomalyDetectorOneClassSVM import AnomalyDetectorOneClassSVM
from anomaly_detector.AnomalyDetectorVanillaNF import AnomalyDetectorVanillaNF
from anomaly_detector.AnomalyDetectorPSCAL import AnomalyDetectorPSCAL

K = 1000
N = 500
anomaly_ratio = 0.05

ad = []
ad.append(AnomalyDetectorCluster())
ad.append(AnomalyDetectorIsoForest())
ad.append(AnomalyDetectorOneClassSVM())
ad.append(AnomalyDetectorLocalOutlierFactor())
#ad.append(AnomalyDetectorAutoencoder())
ad.append(AnomalyDetectorVanillaNF())
ad.append(AnomalyDetectorPSCAL())

n_training_sets =7
n_severities = 5 

severity = np.arange(n_severities)
for i in range(n_training_sets): 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for d in ad:
        name = d.get_name()
        accs = np.zeros(n_severities)
        rocs = np.zeros(n_severities)
        for s in range(n_severities):
            d.init(str(i)+"_"+str(s))
            accuracy, auroc = d.train_eval()
            accs[s] = accuracy
            rocs[s] = auroc
            print(f'{name} Accuracy: {accuracy * 100:.2f}% AUROC: {auroc * 100:.2f}%')
        ax1.plot(severity, accs, label=name)
        ax1.set_xlabel("severity")
        ax1.set_ylabel("Val Acc")
        ax1.legend()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(severity, rocs, label=name)
        ax2.set_xlabel("severity")
        ax2.set_ylabel("AUROC")
        ax2.legend()
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))



    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot plt.show()
    # Saves as a PNG file
    plt.savefig("test_results/results_set_"+str(i)+".png")  

