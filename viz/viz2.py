import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = [0, 1, 2, 3, 4]
WINDOW = 1

# Create a 3x2 subplot layout
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

# Flatten the axes for easier indexing
axs = axs.flatten()

# Define the algorithm list
alg_list = ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]

# Define a consistent color palette
palette = sns.color_palette("tab10", 3)

handles, labels = None, None

for i, d in enumerate(dataset):
    # Read dataset
    data = pd.read_csv(f'res_files_new_eps/{d}.csv')

    # Filter for selected algorithms
    data = data[data['alg'].isin(alg_list)]
    labels = ["Vanilla NF", "Our method", "Vanilla NF (w/o noise)"]
    # Apply rolling average if WINDOW > 1
    data['auroc'] = data['auroc'].rolling(window=WINDOW).mean()
    data = data.dropna().reset_index(drop=True)

    # Plot AUROC
    plot = sns.lineplot(
        data=data,
        x='severity',
        y='auroc',
        hue='alg',
        palette=palette,
        ax=axs[i],
        legend=(i == 0)  # Show legend only on the first plot
    )

    # Capture legend handles and labels from the first plot
    if i == 0:
        handles, labels = axs[i].get_legend_handles_labels(
        )  # Extract from the first subplot
        labels = ["Ratio", "Vanilla NF", "Our method", "Vanilla NF (w/o noise)"]

    axs[i].set_title(f"Dataset {d}")
    axs[i].set_ylabel("AUROC")
    axs[i].set_xlabel("Noise severity")

# Add legend in the last subplot
axs[-1].legend(handles, labels, title="Algorithms", loc='center', fontsize=12)

axs[-1].axis("off")  # Hide axis for the legend subplot

plt.tight_layout()
plt.savefig("tmp_imgs2/combined_plot_auroc3.png")
plt.savefig("tmp_imgs2/combined_plot_auroc3.svg")
