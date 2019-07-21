import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import os

class ReacherPushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(os.path.realpath(__file__)) + '/assets/reacher_push.xml', 2)

    def step(self, a):
        box_vec = self.get_body_com("box0") - self.get_body_com("target")
        finger_vec = self.get_body_com("fingertip") - self.get_body_com("box0")
        reward_dist = - 10 * np.linalg.norm(box_vec) - 0.5 * np.linalg.norm(finger_vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        for _ in range(1000):
            qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.model.nq)# + self.init_qpos
            box_pos = self.np_random.uniform(low=-0.15, high=0.15, size=2)
            qpos[-3] = self.np_random.uniform(low=-0.2, high=0.2, size=1)
            qpos[-2] = self.np_random.uniform(low=0.03, high=0.05, size=1)

            self.goal = self.np_random.uniform(low=-.2, high=.2, size=1)
            self.model.body_pos[-1, -3] = self.goal
            self.model.body_pos[-1, -1] = 0.03
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            qvel[-3:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0:
                break
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.model.body_pos[-1, (-3, -1)],    # target's (x, z) coords
            self.sim.data.qvel.flat,
        ])
