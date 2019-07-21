import torch
from torch import nn, optim
from torch.nn import functional as F

import util

class ActionDecoder(nn.Module):
    def __init__(self, layers, embed_size, traj_len, action_space):
        super().__init__()
        self.traj_len = traj_len
        self.action_space = action_space
        self.embed_dim = embed_size
        self.max_embedding = None

        traj_size = util.prod(self.action_space.shape) * self.traj_len

        self.layers = nn.ModuleList([
            nn.Linear(embed_size, 400),
            *[nn.Linear(400, 400) for _ in range(layers)]
        ])
        self.out_layer = nn.Linear(400, traj_size)

    def forward(self, embedding):
        current = embedding
        for layer in self.layers:
            current = F.selu(layer(current))
        result = F.tanh(self.out_layer(current))
        result = result.view(-1, self.traj_len, *self.action_space.shape)
        return result
