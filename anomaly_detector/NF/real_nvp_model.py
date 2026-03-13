import torch.nn as nn
import torch
import numpy as np


class RealNVP(nn.Module):

    def __init__(self, dim, hidden_dim, num_blocks):
        super(RealNVP, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            [self._create_block() for _ in range(num_blocks)])
        self.permutations = [torch.randperm(dim) for _ in range(num_blocks)]

    def _create_block(self):
        return nn.ModuleDict({
            "s":
            nn.Sequential(
                nn.Linear(self.dim // 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.dim // 2),
                nn.Tanh(),
            ),
            "t":
            nn.Sequential(
                nn.Linear(self.dim // 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.dim // 2),
            ),
        })

    def forward(self, x: torch.Tensor):
        z = x
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)
        for i in range(self.num_blocks):
            z = z[:, self.permutations[i]]
            x1, x2 = z.chunk(2, dim=1)
            s = self.blocks[i]["s"](x1)
            t = self.blocks[i]["t"](x1)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            z = torch.cat([y1, y2], dim=1)

            log_det_jacobian += s.sum(dim=1)
        u = torch.sigmoid(z)
        return u, log_det_jacobian

    @torch.no_grad
    def inverse(self, u):
        z = torch.log(u / (1 - u))
        for i in reversed(range(self.num_blocks)):
            y1, y2 = z.chunk(2, dim=1)
            s = self.blocks[i]["s"](y1)
            t = self.blocks[i]["t"](y1)
            x1 = y1
            x2 = (y2 - t) * torch.exp(-s)
            z = torch.cat([x1, x2], dim=1)
            z = z[:, torch.argsort(self.permutations[i])]
        x = z
        return x

    def num_params(self) -> int:
        n = 0
        for param in self.parameters():
            n += np.prod(param.shape)
        return n
