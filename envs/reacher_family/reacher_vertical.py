import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

class ReacherVerticalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(os.path.realpath(__file__)) + '/assets/reacher_vertical.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
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
            qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.model.nq)
            while True:
                self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(self.goal) < 2:
                    break

            self.model.body_pos[-1, -3] = self.goal[0]
            self.model.body_pos[-1, -1] = self.goal[1] + 0.25
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
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
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
