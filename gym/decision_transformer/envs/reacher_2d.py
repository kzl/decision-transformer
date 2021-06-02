import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import os


class Reacher2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    def __init__(self):
        self.fingertip_sid = 0
        self.target_bid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/reacher_2d.xml', 15)
        self.fingertip_sid = self.sim.model.site_name2id('fingertip')
        self.target_bid = self.sim.model.body_name2id('target')
        utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        tip  = self.data.site_xpos[self.fingertip_sid][:2]
        tar  = self.data.body_xpos[self.target_bid][:2]
        dist = np.sum(np.abs(tip - tar))
        reward_dist = 0.  # - 0.1 * dist
        reward_ctrl = 0.0
        reward_bonus = 1.0 if dist < 0.1 else 0.0
        reward = reward_bonus + reward_ctrl + reward_dist
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_bonus=reward_bonus)

    def _get_obs(self):
        theta = self.data.qpos.ravel()
        tip  = self.data.site_xpos[self.fingertip_sid][:2]
        tar  = self.data.body_xpos[self.target_bid][:2]
        return np.concatenate([
            # self.data.qpos.flat,
            np.sin(theta),
            np.cos(theta),
            self.dt * self.data.qvel.ravel(),
            tip,
            tar,
            tip-tar,
        ])

    def reset_model(self):
        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qpos = self.np_random.uniform(low=-2.0, high=2.0, size=self.model.nq)
        qvel = self.init_qvel * 0.0
        while True:
            self.goal = self.np_random.uniform(low=-1.5, high=1.5, size=2)
            if np.linalg.norm(self.goal) <= 1.0 and np.linalg.norm(self.goal) >= 0.5:
                break
        self.set_state(qpos, qvel)
        self.model.body_pos[self.target_bid][:2] = self.goal
        self.sim.forward()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 5.0
