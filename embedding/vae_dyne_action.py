import torch
from torch import nn
from torch.nn import functional as F


class ActionDynEVAE(nn.Module):
    def __init__(self, layers, traj_size, embed_size, input_nelement):
        super().__init__()
        self.traj_size = traj_size
        self.input_nelement = input_nelement
        self.hidden_size = 400

        self.encoder_layers = nn.ModuleList([
            nn.Linear(traj_size, self.hidden_size),
            *[nn.Linear(self.hidden_size, self.hidden_size) for _ in range(layers)]
        ])

        self.fc21 = nn.Linear(self.hidden_size, embed_size)
        self.fc22 = nn.Linear(self.hidden_size, embed_size)

        self.decoder_layers = nn.ModuleList([
            nn.Linear(embed_size + input_nelement, self.hidden_size),
            *[nn.Linear(self.hidden_size, self.hidden_size) for _ in range(layers)]
        ])
        self.fc4 = nn.Linear(self.hidden_size, input_nelement)


    # encode an action sequence into the embedding space
    def encode(self, x):
        x = x.view(-1, self.traj_size)
        for layer in self.encoder_layers:
            x = F.selu(layer(x))
            # x = layer(x)
        mu, log_sigma2 = self.fc21(x), self.fc22(x)

        return mu, log_sigma2


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    # take the current state and an embedded action sequence
    # and predict s_{t+k}
    def decode(self, s, z):
        s = s.contiguous().view(-1, self.input_nelement)
        current = torch.cat([s, z], 1)
        for layer in self.decoder_layers:
            current = F.selu(layer(current))
        return self.fc4(current)


    def forward(self, s, a):
        mu, logvar = self.encode(a)
        z = self.reparameterize(mu, logvar)
        return self.decode(s, z), mu, logvar
