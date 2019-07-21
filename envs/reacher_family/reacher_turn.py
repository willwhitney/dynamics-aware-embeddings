import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import os

class ReacherTurnEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(os.path.realpath(__file__)) + '/assets/reacher_turn.xml', 2)

    def step(self, a):
        vec = self.sim.data.site_xpos[0] - self.get_body_com("target")
        reward_dist = - 10 * np.linalg.norm(vec)
        reward_ctrl = - 0.5 * np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        for _ in range(1000):
            qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            qvel[-1:] = 0

            target_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
            spinner_pos = np.array([0.35, 0.25])
            target_pos = spinner_pos + 0.15 * np.array([np.cos(target_angle), np.sin(target_angle)])
            self.model.body_pos[-1, (-3, -1)] = target_pos

            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0:
                break
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        # use the xpos of the tip instead of the hinge qpos
        # because when the spinner spins, the qpos will go arbitrarily large
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.body_pos[-1, (-3, -1)],    # target's (x, z) coords
            self.sim.data.qvel.flat,
            self.sim.data.site_xpos[0, (0,2)], # cartesian position of tip of spinner
        ])
