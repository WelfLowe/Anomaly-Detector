from abc import ABC, abstractmethod
from typing import List

import numpy as np

import torch
import anomaly_detector.NF.utils as utl
import anomaly_detector.NF.model as mdl
import anomaly_detector.NF.real_nvp_model as nvp_mdl
import anomaly_detector.NF.config as c


def train_one_epoch(model, optimizer, dataloader, epoch):
    model.train()
    print('.')
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(c.device)

        # Forward pass
        z = model(data)
        jac = model.nf.jacobian(run_forward=False)
        loss = 0.5 * torch.sum(z**2, dim=(1, )) - jac  # Loss for each sample
        mean_loss = torch.mean(loss)  # Mean loss for batch

        # Backward pass and optimization step
        mean_loss.backward()
        optimizer.step()


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


def train(data_holder, epochs, n_coupling_blocks, clamp_alpha, fc_internal,
          dropout, learning_rate, batch_size):
    n_features = data_holder.get_n_features()
    model = mdl.DifferNet(n_features, n_coupling_blocks, clamp_alpha,
                          fc_internal, dropout)
    model.to(c.device)
    optimizer = torch.optim.Adam(
        model.nf.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-5,
    )

    for epoch in range(epochs):
        dataloader = torch.utils.data.DataLoader(
            utl.CustomDataset(data_holder.get_data()),
            batch_size=batch_size,
            shuffle=True,
        )

        train_one_epoch(model, optimizer, dataloader, epoch)
    return model


data = np.load("testset_giustina/train_giustina.npy")
K = data.shape[0]
N = data.shape[1]

# Normalize the data
data_norm = (data - data.min()) / (data.max() - data.min())
data_norm = data_norm.reshape(K, N)
data_holder = utl.DataHolder(data_norm)
print(data_norm.shape)

model = train(data_holder, 200, c.n_coupling_blocks, c.clamp_alpha,
              c.fc_internal, c.dropout, c.learning_rate, c.batch_size)

torch.save(model, "giustina2_vanilla.pth")
