import argparse
import os
import os.path as path
import random
import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import util
import gym_dataset
from action_decoder import ActionDecoder
from vae_dyne_action import ActionDynEVAE

import sys
sys.path.insert(0, '../envs')
import reacher_family

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='default')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100000000, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--kl', type=float, default=1e-4)
parser.add_argument('--norm-loss', type=float, default=1e-4)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--dec-layers', type=int, default=1)
parser.add_argument('--traj-len', type=int, default=4)
parser.add_argument('--env', default='ReacherVertical-v2')
parser.add_argument('--embed-every', type=int, default=10000)
parser.add_argument('--decoder-epochs', type=int, default=10)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.decoder = True

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


result_path = path.join('results', args.env, args.name)
render_path = path.join(result_path, 'render')
os.makedirs(render_path, exist_ok=True)
util.write_options(args, result_path)

# builds a dataset by stepping a gym env with random actions
dataset = gym_dataset.load_or_generate(args.env, args.traj_len,
        qpos_only=False,
        qpos_qvel=True,)
workers = 0

train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)


epoch_size = 10000
train_iterator = iter(train_loader)

# calculate the sizes of everything
action_space = dataset.env.action_space
input_nelement = util.prod(dataset.get_obs().shape)
action_size = util.prod(action_space.shape)
traj_size = action_size * args.traj_len
embed_size = action_size


model = ActionDynEVAE(args.layers, traj_size, embed_size, input_nelement).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    likelihood = F.mse_loss(recon_x, x.view(-1, input_nelement),
            size_average=False)

    # see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # scale the likelihood term by number of state dimensions to make this loss
    # invariant to the environment's observation space
    return likelihood / input_nelement, KLD


def train(epoch):
    model.train()
    train_loss = 0
    train_likelihood = 0
    train_kld = 0
    qpos_loss, qvel_loss = 0, 0
    for batch_idx in range(epoch_size // args.batch_size):
        (states, actions) = next(train_iterator)
        states = states.to(device).float()
        actions = actions.to(device).float()

        pred_states, mu, logvar = model(states[:, 0], actions)
        likelihood, kld = loss_function(pred_states, states[:, -1], mu, logvar)
        qpos_loss += F.mse_loss(pred_states[:, :input_nelement // 2], states[:, -1, :input_nelement // 2])
        qvel_loss += F.mse_loss(pred_states[:, input_nelement // 2:], states[:, -1, input_nelement // 2:])
        loss = likelihood + args.kl * kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_likelihood += likelihood.item()
        train_kld += kld.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(states), epoch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(states)))

    print(('====> Epoch: {} Average loss: {:.4f}'
           '\tLL: {:.6f}\tKLD: {:.6f}').format(
          epoch, train_loss / epoch_size,
          train_likelihood / epoch_size,
          train_kld / epoch_size))
    # print("qpos loss: {:.6f}, qvel_loss: {:.6f}".format(
    #     qpos_loss / epoch_size, qvel_loss / epoch_size))


# sample from the marginal distribution of the encoder instead of the prior
def sample_z_batch():
    (states, actions) = next(train_iterator)
    actions = actions.to(device).float()
    z = model.encode(actions)[0]
    return z


# used to whiten the latent space before training the decoder
def marginal_stats():
    zs = []
    for _ in range(1000):
        (states, actions) = next(train_iterator)
        actions = actions.to(device).float()
        zs.append(model.encode(actions)[0])
    zs = torch.cat(zs, dim=0)
    mean, std = zs.mean(dim=0), zs.std(dim=0)
    white_zs = (zs - mean) / std
    white_max = white_zs.abs().max()
    return mean, std, white_max


# min || E(a) - E(D(E(a))) || + lambda * || D(E(a)) ||
def build_decoder():
    decoder = ActionDecoder(
            args.dec_layers,
            embed_size,
            args.traj_len,
            action_space).to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    dec_epoch_size = 1000

    z_stats = marginal_stats()
    for epoch in range(args.decoder_epochs):
        decoder_loss = 0
        decoder_recon_loss = 0
        decoder_norm_loss = 0

        for batch_idx in range(dec_epoch_size):
            z = sample_z_batch()
            z = (z - z_stats[0].detach()) / z_stats[1].detach()
            decoded_action = decoder(z)
            z_hat = model.encode(decoded_action)[0]
            z_hat_white = (z_hat - z_stats[0].detach()) / z_stats[1].detach()

            recon_loss = F.mse_loss(z_hat_white, z)
            norm_loss = decoded_action.norm(dim=2).sum()
            loss = recon_loss + args.norm_loss * norm_loss

            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()

            decoder_loss += loss.item()
            decoder_recon_loss += recon_loss.item()
            decoder_norm_loss += norm_loss.item()

        print((
            'ActionDecoder epoch: {}\tAverage loss: {:.4f}'
            '\tRecon loss: {:.6f}\tNorm loss: {:.6f}'
        ).format(
          epoch, decoder_loss / (dec_epoch_size * args.batch_size),
          decoder_recon_loss / (dec_epoch_size * args.batch_size),
          decoder_norm_loss / (dec_epoch_size * args.batch_size)))

    # z_stats[2] is the max
    decoder.max_embedding = z_stats[2]
    return decoder


def save_decoder():
    decoder = build_decoder()
    torch.save(decoder, path.join(result_path, 'decoder.pt'))
    return decoder

def save_model():
    torch.save(model, path.join(result_path, 'model.pt'))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)

    # always save at the end of training
    save_model()
    save_decoder()
