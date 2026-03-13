import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(42)

plt.figure(figsize=(16, 8))

for j in range(5):
    # Load the data and labels
    sev = np.random.randint(5, 12)
    run = np.random.randint(0, 9)
    sample_size = 3
    signature = f'{j}_{sev}_{run}'
    data = np.load(f"testsets/train_{signature}.npy")  # Shape: (1000, 500)
    data = (data - data.min()) / (data.max() - data.min())
    labels = np.load(
        f"testsets/train_{signature}_labels.npy")  # Shape: (1000,)

    class_an = data[labels == 1]
    class_no = data[labels == 0]

    # Define colors for the two classes
    class_0_color = "#1f77b4"  # Seaborn blue
    class_1_color = "#ff7f0e"  # Seaborn orange
    r = np.random.choice(range(class_no.shape[0]), size=sample_size)
    for i in r:  #range(class_no.shape[0]):
        plt.subplot(2, 3, j + 1)
        sns.lineplot(data[i], color=class_0_color,
                     alpha=0.5)  #, label='Nominal')

    plt.title(f'Dataset {j}, with severity parameter {sev}')

    r = np.random.choice(range(class_an.shape[0]), size=sample_size)

    for i in r:  #range(class_an.shape[0]):
        plt.subplot(2, 3, j + 1)
        sns.lineplot(data[i], color=class_1_color,
                     alpha=0.5)  #, label='Anomaly')

    #plt.xlabel("Time Step")
    legend_patches = [
        mpatches.Patch(color=class_0_color, label="Nominal"),
        mpatches.Patch(color=class_1_color, label="Anomaly")
    ]
    #plt.legend(handles=legend_patches, loc="upper right")
    #plt.ylabel("Value")
    #plt.title("Noisy dataset of generated signals")
#plt.subplot(2, 3, 6)
#for i in r:  #range(class_an.shape[0]):
#    plt.plot(data[i], color=class_1_color, alpha=0.3, label='Anomaly')

#legend_patches = [
#    mpatches.Patch(color=class_0_color, label="Nominal"),
#    mpatches.Patch(color=class_1_color, label="Anomaly")
#]
#plt.legend(handles=legend_patches, loc="upper right")
plt.savefig('data.png')
