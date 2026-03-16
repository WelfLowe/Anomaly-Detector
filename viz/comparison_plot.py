import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATASETS = [0, 1, 2, 3, 4]
WINDOW = 1

# NOTE: Define your folders, their display tag, and the algorithms to extract
SOURCES = [
    {"dir": "res_files", "tag": "-(eps=0.01)", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    {"dir": "res_files/20260315_234620", "tag": "-(eps=0.75)", "algs": ["PSCAL"]},
    {"dir": "res_files/20260316_000813", "tag": "-(eps=0.0)", "algs": ["PSCAL"]},
    {"dir": "res_files/20260316_072519", "tag": "-(eps=0.01, xi=2)", "algs": ["PSCAL"]},   
]


BASE_LABELS = {
    "vanillaNF": "Vanilla NF",
    "PSCAL": "Our method",
    "vanillaNFnoNoise": "Vanilla NF (w/o noise)"
}

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.flatten()

def load_and_merge_data(dataset_id):
    df_list = []
    for src in SOURCES:
        try:
            df = pd.read_csv(f"{src['dir']}/{dataset_id}.csv")
            df = df[df['alg'].isin(src['algs'])].copy()
            
            # Map to readable names and append the source tag for the legend
            df['alg'] = df['alg'].map(BASE_LABELS).fillna(df['alg']) + src['tag']
            df_list.append(df)
        except FileNotFoundError:
            continue
            
    if not df_list:
        return pd.DataFrame()

    combined_data = pd.concat(df_list, ignore_index=True)
    
    # NOTE: Group by algorithm before rolling to prevent crossover pollution
    combined_data['auroc'] = combined_data.groupby('alg')['auroc'].transform(
        lambda x: x.rolling(window=WINDOW).mean()
    )
    return combined_data.dropna().reset_index(drop=True)


total_lines = sum(len(src['algs']) for src in SOURCES)
palette = sns.color_palette("tab10", total_lines)

for i, d in enumerate(DATASETS):
    data = load_and_merge_data(d)
    if data.empty:
        continue

    sns.lineplot(
        data=data,
        x='severity',
        y='auroc',
        hue='alg',
        palette=palette,
        ax=axs[i],
        legend=(i == 0) 
    )

    axs[i].set_title(f"Dataset {d}")
    axs[i].set_ylabel("AUROC")
    axs[i].set_xlabel("Noise severity")

if axs[0].get_legend():
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, title="Algorithms", loc='center', fontsize=12)
    axs[0].get_legend().remove()

axs[-1].axis("off")

plt.tight_layout()
plt.savefig("tmp/full_plot.png")