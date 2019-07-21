import torch
from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self, state_embed_size, hidden_size):
        super().__init__()
        self.state_embed_size = state_embed_size
        self.hidden_size = hidden_size

        self.decoder_lins = nn.Sequential(
            nn.Linear(state_embed_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(True),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(True),
        )

        self.decoder_generator = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, 32 * 8, 4, 1, 0),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(True),
            # state size. (32*8) x 4 x 4
            nn.ConvTranspose2d(32 * 8, 32 * 4, 4, 2, 1),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True),
            # state size. (32*4) x 8 x 8
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True),
            # state size. (32*2) x 16 x 16
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (32) x 32 x 32

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            # nn.ConvTranspose2d(32, 3 * 2, 4, 2, 1),

            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, state_z):
        state_z = state_z.contiguous().view(-1, self.state_embed_size)

        current = self.decoder_lins(state_z)
        current = current.reshape(*current.shape, 1, 1)
        current = self.decoder_generator(current)
        return current

class RegularVAE(nn.Module):
    def __init__(self, state_embed_size):
        super().__init__()
        self.state_embed_size = state_embed_size
        self.hidden_size = 400

        self.state_encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            # state size. (32) x 32 x 32
            nn.Conv2d(32, 32 * 2, 4, 2, 1),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(inplace=True),
            # state size. (32*2) x 16 x 16
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(inplace=True),
            # state size. (32*4) x 8 x 8
            nn.Conv2d(32 * 4, 32 * 8, 4, 2, 1),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(inplace=True),
            # state size. (32*8) x 4 x 4
            nn.Conv2d(32 * 8, self.hidden_size, 4, 1, 0),
            nn.ReLU(inplace=True),
        )

        self.lin_state_mu = nn.Linear(self.hidden_size, self.state_embed_size)
        self.lin_state_sigma = nn.Linear(self.hidden_size, self.state_embed_size)

        self.decoder = Decoder(self.state_embed_size, self.hidden_size)

    def encode_state(self, x):
        x = self.state_encoder(x).reshape(x.size(0), -1)
        return self.lin_state_mu(x), self.lin_state_sigma(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def sample(self, n=1):
        with torch.no_grad():
            state_noise = torch.Tensor(n, self.state_embed_size).normal_().cuda()

            return self.decode(state_noise)

    def forward(self, s):
        state_mu, state_logvar = self.encode_state(s)
        state_z = self.reparameterize(state_mu, state_logvar)

        return self.decode(state_z), state_mu, state_logvar

    def decode(self, state_z):
        decoded = self.decoder(state_z)
        return decoded
