import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import utils

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def build_conv(arch, img_width, stack=3):
    if arch == "ilya":
        # architecture used in https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        # by Ilya Kostrikov
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 32, 8, stride=4),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.Conv2d(32, 32, 3),
        ])

    elif arch == "ilya_bn":
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
        ])

    elif arch == "impala":
        conv_layers = nn.ModuleList([
            nn.Conv2d(stack * 3, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])

    elif arch == "impala_bn":
        conv_layers = nn.ModuleList([
            nn.BatchNorm2d(stack * 3),
            nn.Conv2d(stack * 3, 16, 8, stride=4),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 4, stride=2),
        ])

    conv_output_dim = utils.prod(utils.conv_list_out_dim(conv_layers, img_width, img_width))
    return conv_layers, conv_output_dim




class Actor(nn.Module):
    def __init__(self, action_dim, max_action, conv_layers, conv_output_dim, img_width, stack):
        super(Actor, self).__init__()
        self.conv_layers, self.conv_output_dim = conv_layers, conv_output_dim

        self.lin_layers = nn.ModuleList([
            nn.BatchNorm1d(self.conv_output_dim),
            nn.Linear(self.conv_output_dim, 200),
            nn.BatchNorm1d(200),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.Linear(200, action_dim),
        ])

        self.max_action = max_action

    def forward(self, x):
        x = x.view(-1, self.conv_output_dim)

        for i, layer in enumerate(self.lin_layers):
            x = layer(x)
            if i < (len(self.lin_layers) - 1) and isinstance(layer, nn.Linear):
                x = F.relu(x)

        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, conv_layers, conv_output_dim, img_width, stack):
        super(Critic, self).__init__()
        self.conv_layers, self.conv_output_dim = conv_layers, conv_output_dim

        # repeat the action to make it similar in dimension to the state
        # otherwise it is ignored with high-dimensional states
        self.action_repeat = self.conv_output_dim // action_dim
        action_dim = action_dim * self.action_repeat

        self.q1_lin_layers = nn.ModuleList([
            nn.Linear(self.conv_output_dim + action_dim, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 1),
        ])

        self.q2_lin_layers = nn.ModuleList([
            nn.Linear(self.conv_output_dim + action_dim, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 1),
        ])


    def forward(self, x, u):
        return self.Q1(x, u), self.Q2(x, u)


    def Q1(self, x, u):
        x = x.view(-1, self.conv_output_dim)
        x = torch.cat([x, u.repeat([1, self.action_repeat])], dim=1)
        for i, layer in enumerate(self.q1_lin_layers):
            x = layer(x)
            if i < (len(self.q1_lin_layers) - 1) and isinstance(layer, nn.Linear): x = F.relu(x)

        return x

    def Q2(self, x, u):
        x = x.view(-1, self.conv_output_dim)
        x = torch.cat([x, u.repeat([1, self.action_repeat])], dim=1)
        for i, layer in enumerate(self.q2_lin_layers):
            x = layer(x)
            if i < (len(self.q2_lin_layers) - 1) and isinstance(layer, nn.Linear): x = F.relu(x)

        return x


class PixelTD3(object):
    def __init__(self, state_dim, action_dim, max_action, arch="ilya_bn", img_width=128, stack=4):
        self.conv_layers, conv_output_dim = build_conv(arch, img_width, stack)
        self.target_conv_layers, _ = build_conv(arch, img_width, stack)

        self.actor = Actor(action_dim, max_action, self.conv_layers, conv_output_dim, img_width, stack).to(device)
        self.actor_target = Actor(action_dim, max_action, self.target_conv_layers, conv_output_dim, img_width, stack).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(action_dim, self.conv_layers, conv_output_dim, img_width, stack).to(device)
        self.critic_target = Critic(action_dim, self.target_conv_layers, conv_output_dim, img_width, stack).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # print(self.actor)
        # print(self.critic)
        # print("Actor params: ", sum([p.nelement() for p in self.actor.parameters()]))

        self.max_action = max_action

    def conv_forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear): x = F.relu(x)
        return x

    def target_conv_forward(self, x):
        for layer in self.target_conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear): x = F.relu(x)
        return x

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = self.conv_forward(state)
        action = self.actor(state)
        action = action.cpu()
        action = action.data.numpy()
        action = action.flatten()
        return action

    def reset(self):
        pass

    def mode(self, mode):
        if mode == 'eval':
            self.actor.eval()
        elif mode == 'train':
            self.actor.train()
        else:
            assert False

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.mode('train')

        loader = DataLoader(replay_buffer, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        it = 0
        while it < iterations:
            for batch in loader:
                it += 1
                if it >= iterations: break
                raw_state, raw_next_state, action, reward, done = batch

                raw_state = raw_state.to(device)
                state = self.conv_forward(raw_state)

                raw_next_state = raw_next_state.to(device)
                target_next_state = self.target_conv_forward(raw_next_state)

                action = action.to(device)
                reward = reward.float().unsqueeze(1).to(device)
                done = (1 - done).float().unsqueeze(1).to(device)

                # Select action according to policy and add clipped noise
                noise = torch.FloatTensor(action.size()).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(target_next_state) + noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(target_next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q).detach()

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Delayed policy updates
                if it % policy_freq == 0:
                    # only update representation based on critic (performs better)
                    state = state.detach()

                    # Compute actor loss
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                    # Optimize the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.mode('eval')


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self, '%s/%s_all.pth' % (directory, filename))


    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location='cpu'))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu'))
        else:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

def load(filename, directory):
    if not torch.cuda.is_available():
        return torch.load('%s/%s_all.pth' % (directory, filename), map_location='cpu')
    else:
        return torch.load('%s/%s_all.pth' % (directory, filename))
