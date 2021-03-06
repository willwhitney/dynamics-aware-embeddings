import numpy as np
from gym.core import Wrapper
import skimage.transform

import torch
from gym import spaces


class EmbedStackPixelObservationWrapper(Wrapper):
    def __init__(self, env, encoder, stack=4, img_width=64, source_img_width=64):
        self.env = env
        self.encoder = encoder
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(shape=(self.encoder.state_embed_size,), low=-np.inf, high=np.inf)
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.img_width = img_width
        self.source_img_width = source_img_width

        self.stack = stack
        self.imgs = [np.zeros([3, self.img_width, self.img_width]) for _ in range(self.stack)]

    def render_obs(self, color_last=False):
        raw_img = self.env.render(mode='rgb_array', height=self.source_img_width, width=self.source_img_width)
        resized = skimage.transform.resize(raw_img, (self.img_width, self.img_width))
        if color_last: return resized
        else: return resized.transpose([2, 0, 1])

    def render(self, *args, **kwargs):
        return self.render_obs(color_last=True) * 255

    def observation(self):
        with torch.no_grad():
            stacked_imgs = np.concatenate(self.imgs, axis=0)
            img_input = torch.tensor(stacked_imgs).float().unsqueeze(0).cuda()
            embedded_imgs = self.encoder.encode_state(img_input)[0][0].cpu().numpy()
        return embedded_imgs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        img = self.render_obs()
        self.imgs.pop(0)
        self.imgs.append(img)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.imgs = [np.zeros([3, self.img_width, self.img_width]) for _ in range(self.stack - 1)] + [self.render_obs()]
        for _ in range(self.stack - 1):
            self.step(self.action_space.sample())
        self.env._elapsed_steps = 0
        return self.observation()
