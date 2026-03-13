import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = [0, 1, 2, 3, 4]
WINDOW = 1
for d in dataset:
    data = pd.read_csv(f'res_files_new/{d}.csv')

    alg_list = [
        "Cluster", "vanillaNF", "PSCAL", "Isolation Forest", "vanillaNFnoNoise"
    ]

    # Set WINDOW=1 for standard plot otherwise rolling average with window=WINDOW
    data = data[data['alg'].isin(alg_list)]

    data['acc'] = data['acc'].rolling(window=WINDOW).mean()
    data['auroc'] = data['auroc'].rolling(window=WINDOW).mean()
    data = data.dropna().reset_index(drop=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.lineplot(
        data=data,
        x='severity',
        y='acc',
        hue='alg',
        #errorbar=('ci', 95),
        ax=axs[0])
    axs[0].set_title("Accuracy by Severity")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Severity")

    sns.lineplot(data=data, x='severity', y='auroc', hue='alg', ax=axs[1])
    axs[1].set_title("AUROC by Severity")
    axs[1].set_ylabel("AUROC")
    axs[1].set_xlabel("Severity")

    plt.tight_layout()
    plt.savefig("tmp_imgs2/" + str(d) + ".png")
    plt.show()
