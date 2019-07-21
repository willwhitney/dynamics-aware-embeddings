import argparse
import os
import os.path as path
import random
import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision.utils as tvu

import util
import gym_dataset
from vae_regular import RegularVAE

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

parser.add_argument('--dataset-size', type=int, default=100000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--state-kl', type=float, default=5e-7)
parser.add_argument('--state-embed-size', type=int, default=25)
parser.add_argument('--env', default='gridworld')
parser.add_argument('--embed-every', type=int, default=10000)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.decoder = True

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


result_path = path.join('results', 'Pixel' + args.env, args.name)
render_path = path.join(result_path, 'render')
os.makedirs(render_path, exist_ok=True)
util.write_options(args, result_path)


dataset = gym_dataset.load_or_generate(args.env, args.traj_len,
        cache_size=args.dataset_size,
        qpos_only=False,
        qpos_qvel=False,
        delta=False,
        whiten=False,
        pixels=True)

train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)


epoch_size = 10000
train_iterator = iter(train_loader)

input_nelement = util.prod(dataset.get_obs().shape)

model = RegularVAE(args.state_embed_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, state_mu, state_logvar):
    # BCE tends to work better on images
    likelihood = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    state_KLD = -0.5 * torch.sum(1 + state_logvar - state_mu.pow(2) - state_logvar.exp())

    # scale the likelihood term by number of state dimensions to make this loss
    # invariant to the environment's observation space
    return likelihood / input_nelement, state_KLD


# sawtooth KL annealing schedule inspired by https://arxiv.org/abs/1903.10145
def kl_schedule(epoch):
    if epoch < 400:
        return ((epoch - 1) % 100) / 100
    else:
        return 1

def train(epoch):
    model.train()
    train_loss = 0
    train_likelihood = 0
    train_state_kld = 0
    for batch_idx in range(epoch_size // args.batch_size):
        (states, actions) = next(train_iterator)
        states = states.to(device).float()
        actions = actions.to(device).float()

        input_states = states[:, 0, :3]
        pred_states, state_mu, state_logvar = model(input_states)


        likelihood, state_kld = loss_function(pred_states, input_states, state_mu, state_logvar)
        loss = likelihood + kl_schedule(epoch) * (args.state_kl * state_kld)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_likelihood += likelihood.item()
        train_state_kld += state_kld.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print(('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}').format(
                    epoch, batch_idx * len(states), epoch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(states)))

    # render reconstructions
    for i, inputs, pred in zip(range(min(5, len(pred_states))), input_states, pred_states):
        inputs = list(inputs.reshape([-1, 3, 64, 64]))
        img_path = path.join(render_path, "recon{}_{}.png".format(epoch, i))
        util.save_image(img_path, torch.cat([*inputs, pred], dim=2).detach())

    # render some example observations
    if epoch == 1:
        for i in range(min(5, len(states))):
            imgs = states[i]
            img_path = path.join(render_path, "observations{}_{}.png".format(epoch, i))
            imgs = imgs.reshape([-1, 3, 64, 64])
            tvu.save_image(imgs, img_path, nrow=4)

    print(('====> Epoch: {} Average loss: {:.4f}'
           '\tLL: {:.6f}\tstate KLD: {:.6f}').format(
                epoch, train_loss / epoch_size,
                train_likelihood / epoch_size,
                train_state_kld / epoch_size,))


def render_samples(epoch):
    samples = model.sample(5)
    for i, sample in enumerate(samples):
        img_path = path.join(render_path, "samples{}_{}.png".format(epoch, i))
        util.save_image(img_path, sample)

def save_model(epoch):
    torch.save(model, path.join(result_path, 'model_{}.pt'.format(epoch)))
    torch.save(model, path.join(result_path, 'model.pt'))

if __name__ == "__main__":
    save_model(0)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % 10 == 0: render_samples(epoch)
        if epoch % args.embed_every == 0: save_model(epoch)

    # always save at the end of training
    save_model(args.epochs)
