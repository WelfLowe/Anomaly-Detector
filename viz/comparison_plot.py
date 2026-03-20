import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATASETS = [0, 1, 2, 3, 4]
WINDOW = 1

SOURCES = [
    #{"dir": "res_files/original", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260316_072519", "algs": ["PSCAL"]},
    #{"dir": "res_files/20260316_134641", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260316_150134", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260316_150229", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260316_153137", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260317_003719", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260317_092943", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260317_103859", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    #{"dir": "res_files/20260317_120250", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
    {"dir": "res_files/20260317_142258", "algs": ["vanillaNF", "PSCAL", "vanillaNFnoNoise"]},
]

BASE_LABELS = {
    "vanillaNF": "Vanilla NF",
    "PSCAL": "Our method",
    "vanillaNFnoNoise": "Vanilla NF (w/o noise)"
}

# NOTE: Dynamically generate tags from manifest params
for src in SOURCES:
    manifest_path = os.path.join(src["dir"], "manifest.json")
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        params = manifest["algorithms"][0]["params"]
        x = params.get("std_cutoff", 0)
        e = params.get("explore_eps", 0)
        h = params.get("fc_internal", 0)
        
        src["tag"] = f" (x{x:.1f}-e{e:.1f}-h{h:.1f})"
    except (FileNotFoundError, KeyError, IndexError):
        src["tag"] = " (unknown)"

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.flatten()

def load_and_merge_data(dataset_id):
    df_list = []
    for src in SOURCES:
        file_path = os.path.join(src['dir'], f"{dataset_id}.csv")
        try:
            df = pd.read_csv(file_path)
            df = df[df['alg'].isin(src['algs'])].copy()
            df['alg'] = df['alg'].map(BASE_LABELS).fillna(df['alg'])# + src['tag']
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
        legend=(i == 0),
        errorbar=("ci", 95)
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