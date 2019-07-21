import numpy as np
from gym.core import Wrapper
import skimage.transform
import torch
from gym import spaces

class EmbedPixelObservationWrapper(Wrapper):
    def __init__(self, env, encoder, stack=4, img_width=64, source_img_width=64):
        self.env = env
        self.encoder = encoder
        self.stack = stack
        self.img_width = img_width
        self.source_img_width = source_img_width

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(shape=(self.stack * self.encoder.state_embed_size,), low=-np.inf, high=np.inf)
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.observations = [np.zeros([self.encoder.state_embed_size]) for _ in range(self.stack)]


    def render_obs(self, color_last=False):
        with torch.no_grad():
            raw_img = self.env.render(mode='rgb_array', height=self.source_img_width, width=self.source_img_width)
            resized = skimage.transform.resize(raw_img, (self.img_width, self.img_width))
            resized = resized.transpose([2, 0, 1])
            embedded = self.encoder.encode_state(torch.from_numpy(resized).float().cuda().unsqueeze(0))[0]

            return embedded.cpu().squeeze().numpy()


    def render(self, *args, **kwargs):
        raw_img = self.env.render(mode='rgb_array', height=self.source_img_width, width=self.source_img_width)
        resized = skimage.transform.resize(raw_img, (self.img_width, self.img_width))
        return resized * 255


    def observation(self):
        return np.concatenate(self.observations, axis=0)


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed_obs = self.render_obs()
        self.observations.pop(0)
        self.observations.append(processed_obs)
        return self.observation(), reward, done, info


    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.observations = [np.zeros([self.encoder.state_embed_size])
                             for _ in range(self.stack - 1)] + [self.render_obs()]
        for _ in range(self.stack - 1):
            self.step(self.action_space.sample())
        self.env._elapsed_steps = 0

        return self.observation()
