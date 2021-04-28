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
from action_decoder import ActionDecoder
from vae_dyne_sa import StateActionDynEVAE

import sys
sys.path.insert(0, '../envs')
import reacher_family

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='default')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100000000, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--dataset-size', type=int, default=100000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--state-kl', type=float, default=5e-7)
parser.add_argument('--action-kl', type=float, default=5e-7)
parser.add_argument('--norm-loss', type=float, default=1e-4)
parser.add_argument('--dec-layers', type=int, default=1)
parser.add_argument('--traj-len', type=int, default=4)
parser.add_argument('--state-embed-size', type=int, default=100)
parser.add_argument('--env', default='ReacherVertical-v2')
parser.add_argument('--embed-every', type=int, default=10000)
parser.add_argument('--decoder-epochs', type=int, default=10)

parser.add_argument('--source-img-width', type=int, default=64)
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


# builds a dataset by stepping a gym env with random actions
dataset = gym_dataset.load_or_generate(args.env, args.traj_len,
        cache_size=args.dataset_size,
        qpos_only=False,
        qpos_qvel=False,
        delta=False,
        whiten=False,
        pixels=True,
        source_img_width=args.source_img_width)

train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)


epoch_size = 10000
train_iterator = iter(train_loader)


# calculate the sizes of everything
action_space = dataset.env.action_space
input_nelement = util.prod(dataset.get_obs().shape)
action_size = util.prod(action_space.shape)
action_embed_size = action_size
traj_size = action_size * args.traj_len



model = StateActionDynEVAE(traj_size, action_embed_size, args.state_embed_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, state_mu, state_logvar, action_mu, action_logvar):
    # BCE tends to work better on images
    likelihood = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    state_KLD = -0.5 * torch.sum(1 + state_logvar - state_mu.pow(2) - state_logvar.exp())
    action_KLD = -0.5 * torch.sum(1 + action_logvar - action_mu.pow(2) - action_logvar.exp())

    # scale the likelihood term by number of state dimensions to make this loss
    # invariant to the environment's observation space
    return likelihood / input_nelement, state_KLD, action_KLD

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
    train_action_kld = 0
    for batch_idx in range(epoch_size // args.batch_size):
        (states, actions) = next(train_iterator)
        states = states.to(device).float()
        actions = actions.to(device).float()

        input_states = states[:, 0]
        pred_states, state_mu, state_logvar, action_mu, action_logvar = model(input_states, actions)

        # -6: is the last two images in the sequence
        target_states = states[:, -1, -6:].contiguous()

        likelihood, state_kld, action_kld = loss_function(pred_states, target_states, state_mu, state_logvar, action_mu, action_logvar)
        loss = likelihood + kl_schedule(epoch) * (args.state_kl * state_kld + args.action_kl * action_kld)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_likelihood += likelihood.item()
        train_state_kld += state_kld.item()
        train_action_kld += action_kld.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print(('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}').format(
                    epoch, batch_idx * len(states), epoch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(states)))

    # render reconstructions
    for i, inputs, target, pred in zip(range(min(5, len(pred_states))), input_states, target_states, pred_states):
        inputs = list(inputs.reshape([-1, 3, 64, 64]))
        img_path = path.join(render_path, "recon{}_{}.png".format(epoch, i))
        util.save_image(img_path, torch.cat([*inputs, target[-6:-3], target[-3:], pred[-6:-3], pred[-3:]], dim=2).detach())

    # render some example observations
    if epoch == 1:
        for i in range(min(5, len(states))):
            imgs = states[i]
            img_path = path.join(render_path, "observations{}_{}.png".format(epoch, i))
            imgs = imgs.reshape([-1, 3, 64, 64])
            tvu.save_image(imgs, img_path, nrow=4)

    print(('====> Epoch: {} Average loss: {:.4f}'
           '\tLL: {:.6f}\tstate KLD: {:.6f}\taction KLD: {:.6f}').format(
                epoch, train_loss / epoch_size,
                train_likelihood / epoch_size,
                train_state_kld / epoch_size,
                train_action_kld / epoch_size,))


# sample from the marginal distribution of the encoder instead of the prior
def sample_z_batch():
    (states, actions) = next(train_iterator)
    actions = actions.to(device).float()
    z = model.encode_actions(actions)[0]
    return z


def render_samples(epoch):
    samples = model.sample(5)
    for i, sample in enumerate(samples):
        img_path = path.join(render_path, "samples{}_{}.png".format(epoch, i))
        util.save_image(img_path, torch.cat([sample[:3], sample[3:]], dim=2))


def marginal_stats():
    zs = []
    for _ in range(1000):
        (states, actions) = next(train_iterator)
        actions = actions.to(device).float()
        zs.append(model.encode_actions(actions)[0])
    zs = torch.cat(zs, dim=0)
    mean, std = zs.mean(dim=0), zs.std(dim=0)
    white_zs = (zs - mean) / std
    white_max = white_zs.abs().max()
    return mean, std, white_max


# min || E(a) - E(D(E(a))) || + alpha * || D(E(a)) ||
def build_decoder():
    decoder = ActionDecoder(
            args.dec_layers,
            action_embed_size,
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
            z_hat = model.encode_actions(decoded_action)[0]
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


def save_model(epoch):
    torch.save(model, path.join(result_path, 'model_{}.pt'.format(epoch)))
    torch.save(model, path.join(result_path, 'model.pt'.format(epoch)))

if __name__ == "__main__":
    save_model(0)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % 10 == 0: render_samples(epoch)

        if epoch % args.embed_every == 0:
            save_model(epoch)
            save_decoder()

    # always save at the end of training
    save_model(args.epochs)
    save_decoder()
