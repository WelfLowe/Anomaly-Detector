import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = [0, 1, 2, 3, 4]
WINDOW = 1

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.flatten()
alg_list = ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]
palette = sns.color_palette("tab10", 3)

handles, labels = None, None

for i, d in enumerate(dataset):
    data = pd.read_csv(f'res_files/{d}.csv')
    data = data[data['alg'].isin(alg_list)]
    labels = ["Vanilla NF", "Our method", "Vanilla NF (w/o noise)"]
    data['auroc'] = data['auroc'].rolling(window=WINDOW).mean()
    data = data.dropna().reset_index(drop=True)
    plot = sns.lineplot(
        data=data,
        x='severity',
        y='auroc',
        hue='alg',
        palette=palette,
        ax=axs[i],
        legend=(i == 0)  # Show legend only on the first plot
    )

    if i == 0:
        handles, labels = axs[i].get_legend_handles_labels(
        ) 
        labels = ["Vanilla NF", "Our method", "Vanilla NF (w/o noise)"]

    axs[i].set_title(f"Dataset {d}")
    axs[i].set_ylabel("AUROC")
    axs[i].set_xlabel("Noise severity")

axs[-1].legend(handles, labels, title="Algorithms", loc='center', fontsize=12)
axs[-1].axis("off")

plt.tight_layout()
plt.savefig("tmp/full_plot.png")
#SVG
#plt.savefig("tmp_imgs2/combined_plot_auroc3.svg")
