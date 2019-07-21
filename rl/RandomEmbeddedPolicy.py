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


class RandomEmbeddedPolicy(object):
    def __init__(self, max_action, decoder, max_e_action=None):
        self.decoder = decoder
        self.e_action_dim = decoder.embed_dim
        self.max_action = max_action

        # set the maximum action in embedding space to the largest value the decoder saw during training
        self.max_e_action = max_e_action if max_e_action is not None else float(self.decoder.max_embedding)

        self.pending_plan = torch.Tensor(0, 0, 0).to(device)
        self.current_e_action = None


    def select_action(self, state):
        if self.pending_plan.size(1) == 0:
            e_action = torch.Tensor(1, self.e_action_dim).to(device)
            # e_action.normal_(0, 1)
            e_action.uniform_(-self.max_e_action, self.max_e_action)
            # import ipdb; ipdb.set_trace()

            self.pending_plan = self.decoder(e_action)
            self.current_e_action = e_action

        # next action is head of plan, new plan is tail of current plan
        action = self.pending_plan[:, 0].cpu().data.numpy().flatten()
        self.pending_plan = self.pending_plan[:, 1:]

        # ensure that the decoded action is legal in the environment
        action = action.clip(-self.max_action, self.max_action)

        plan_step = self.decoder.traj_len - self.pending_plan.size(1) - 1
        return action, self.current_e_action[0].detach().cpu().numpy(), plan_step


    def reset(self):
        self.pending_plan = torch.Tensor(0, 0, 0).to(device)
        self.current_e_action = None
