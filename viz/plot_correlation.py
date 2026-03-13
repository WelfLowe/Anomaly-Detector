import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Path to the file
#file_path = "logs/training_log_7_0.csv"
file_path = 'training_log.csv'
data = []

df = pd.read_csv(file_path)
df.columns = ["r", "s", "Epoch", "Batch", "Sample", "Label", "Loss"]


def normalize_loss(group_df):
    scaler = StandardScaler()
    group_df['Normalized_Loss'] = scaler.fit_transform(group_df[['Loss']])
    return group_df


df = df.groupby(['Epoch'], group_keys=False).apply(normalize_loss)

average_loss = df.groupby(['r', 's', 'Epoch',
                           'Label'])['Normalized_Loss'].mean().reset_index()

plt.figure(figsize=(12, 4))
labels = ['Nominal', 'Anomaly']

for i, label in enumerate(average_loss['Label'].unique()):
    label_data = average_loss[average_loss['Label'] == label]
    plt.plot(
        label_data['Epoch'],
        label_data['Normalized_Loss'],
        #marker='.',
        label=labels[i])
    plt.xlim([0, 500])

plt.xlabel('Epoch')
plt.ylabel('Mean normalized loss')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('yes.png')
