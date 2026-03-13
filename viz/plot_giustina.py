import numpy as np
import torch
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.real_nvp_model as nvp_mdl
import anomaly_detector.NF.config as c
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


def find_lists_with_pos_neg_transition(data):
    indices = []

    for i, lst in enumerate(data):
        for j in range(40, min(440,
                               len(lst) -
                               1)):  # Kontrollera att indexet är giltigt
            if lst[j] * lst[
                    j +
                    1] < 0:  # Ett positivt och ett negativt tal bredvid varandra
                indices.append(i)
                break  # Hoppa till nästa lista om vi hittat en match

    return indices


def evaluate(model, dataloader):
    model.eval()
    loss_list = []
    for data, _ in dataloader:
        data = data.to(c.device)
        z = model(data)
        jac = model.nf.jacobian(run_forward=False)
        loss = 0.5 * torch.sum(z**2, dim=(1, )) - jac
        loss_list += list(utl.t2np(loss))
    return loss_list


# Load and normalize data
data = np.load("testset_giustina/train_giustina.npy")
data_norm = (data - data.min()) / (data.max() - data.min())
data_norm = data_norm.reshape(data.shape[0], data.shape[1])
data_holder = utl.DataHolder(data_norm)

# Load model
# 'giustina2_vanilla.pth'
# 'giustina2.pth'


def get_res(model_path):
    model = torch.load(model_path)
    model.to('cuda')

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        utl.CustomDataset(data_holder.get_data()),
        batch_size=c.batch_size,
        shuffle=False,
    )

    # Compute anomaly scores
    scores_val = evaluate(model, dataloader)
    scores_val = np.array(scores_val)

    # Determine anomalies
    anomalies = scores_val > 3 * np.std(scores_val)
    prediction_labels = np.where(anomalies, 0, 1)  # 0 = anomaly, 1 = normal

    # Normalize scores for coloring
    scores_val = (scores_val - np.min(scores_val)) / (np.max(scores_val) -
                                                      np.min(scores_val))

    idx_res = find_lists_with_pos_neg_transition(data[prediction_labels == 0])
    print(idx_res)
    print(scores_val[idx_res])
    print(4 * np.std(scores_val))
    return scores_val, prediction_labels


scores_val, prediction_labels = get_res('giustina2.pth')
scores_val2, prediction_labels2 = get_res('giustina2_vanilla.pth')

# Define colormap
cmap = cm.coolwarm  # Choose a colormap
norm = mcolors.Normalize(vmin=min(scores_val), vmax=max(scores_val))

normal_indices = np.where(prediction_labels == 1)[0]
data_normal = data[normal_indices]

normal_indices = np.where(prediction_labels2 == 1)[0]
data_normal2 = data[normal_indices]

blue_color = cmap(0)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for i in range(data.shape[0]):
    color = cmap(norm(scores_val[i]))  # Map score to color
    axes[0].plot(data[i], color=color, alpha=0.7)

axes[0].set_title("All sequences")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Values")
axes[0].set_ylim([-3.2, 1.5])

for i in range(data_normal.shape[0]):
    axes[1].plot(data_normal[i], color=blue_color, alpha=0.7)

for i in range(data_normal.shape[0]):
    axes[2].plot(data_normal2[i], color=blue_color, alpha=0.7)

axes[1].set_title("Predicted nominal sequences (by threshold $3\sigma$)")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Values")
axes[1].set_ylim([-3.2, 1.5])

axes[2].set_title("Predicted nominal sequences (by threshold $3\sigma$)")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Values")
axes[2].set_ylim([-3.2, 1.5])

# Add colorbar only for the left plot
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[0])
cbar.set_label("Score Value")

# Save the figure
plt.tight_layout()
plt.savefig('all_die_dator_vanilla2.png')
