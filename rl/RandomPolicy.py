import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class RandomPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        action = self.action_space.sample()
        # action.fill(1)
        return action

    def reset(self):
        pass

class ConstantPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        action = self.action_space.sample()
        action.fill(1)
        return action

    def reset(self):
        pass
