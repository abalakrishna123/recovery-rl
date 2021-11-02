'''
Built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import numpy as np
import moviepy.editor as mpy
import copy
import os.path as osp
from env.base_mujoco_env import BaseMujocoEnv
import matplotlib.pyplot as plt
from gym.spaces import Box
import os
import mujoco_py
"""
Constants associated with the Object Extraction (Dynamic Obstacle) env.
"""

FIXED_ENV = False
GT_STATE = True
EARLY_TERMINATION = True


def no_rot_dynamics(prev_target_qpos, action):
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:3] = action[:3] + prev_target_qpos[:3]
    target_qpos[4] = action[3]
    return target_qpos


def clip_target_qpos(target, lb, ub):
    target[:len(lb)] = np.clip(target[:len(lb)], lb, ub)
    return target


class ObjDynamicExtraction(BaseMujocoEnv):
    def __init__(self):
        parent_params = {'viewer_image_height': 480, 'viewer_image_width': 640, 'ncam': 1}
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        self.reset_xml = os.path.join(envs_folder, 'assets/shelf_dynamic.xml')
        super().__init__(self.reset_xml, parent_params)
        self._adim = 4
        self.substeps = 500
        self.low_bound = np.array([-0.4, -0.4, -0.05])
        self.high_bound = np.array([0.4, 0.4, 0.15])
        self.ac_high = np.array([0.03, 0.03, 0.03, 0.1])
        self.ac_low = -self.ac_high
        self.action_space = Box(self.ac_low, self.ac_high)
        self._previous_target_qpos = None
        self._previous_target_qpos_dynamic_obs = None
        self.target_height_thresh = 0.03
        self.object_fall_thresh = -0.03
        self.obj_y_dist_range = np.array([0.05, 0.05])
        self.obj_x_range = np.array([-0.15, -0.07])
        self.randomize_objects = not FIXED_ENV
        self.gt_state = GT_STATE
        self._max_episode_steps = 25

        if self.gt_state:
            self.observation_space = Box(low=-np.inf,
                                         high=np.inf,
                                         shape=(33, ))
        else:
            self.observation_space = (48, 64, 3)
        self.reset()

    def render(self):
        return super().render()[:, ::-1].copy().squeeze(
        )  # cameras are flipped in height dimension

    def reset(self):
        self._reset_sim(self.reset_xml)

        #clear our observations from last rollout
        self._last_obs = None
        self.timestep = 0
        state = self.sim.get_state()
        pos = np.copy(state.qpos[:])
        pos[12:] = self.object_reset_poses().ravel()
        state.qpos[:] = pos
        self.sim.set_state(state)

        self.sim.forward()

        self._previous_target_qpos = copy.deepcopy(
            self.sim.data.qpos[:5].squeeze())
        self._previous_target_qpos[-1] = self.low_bound[-1]
        self._previous_target_qpos_dynamic_obs = copy.deepcopy(
            self.sim.data.qpos[6:11].squeeze())
        self._previous_target_qpos_dynamic_obs[-1] = self.low_bound[-1]
        self.step([0, 0, 0, 0], dynamic_obs_action_in=[0, 0, 0, 0.6])

        if self.gt_state:
            return pos
        else:
            return self.render()

    def step(self, action, dynamic_obs_action_in=None):
        position = self.position
        action = np.clip(action, self.ac_low, self.ac_high)
        target_qpos = self._next_qpos(action)
        if self._previous_target_qpos is None:
            self._previous_target_qpos = target_qpos
        finger_force = np.zeros(2)

        if dynamic_obs_action_in is None:
            if self.timestep < 15:
                dynamic_obs_action = [0.035, 0, 0, 0.6]
            else:
                dynamic_obs_action = [0, 0, 0, 0.6]
        else:
            dynamic_obs_action = dynamic_obs_action_in

        dynamic_obs_action = np.clip(dynamic_obs_action, self.ac_low,
                                     self.ac_high)
        target_qpos_dynamic_obs = self._next_qpos_dynamic_obs(
            dynamic_obs_action)
        if self._previous_target_qpos_dynamic_obs is None:
            self._previous_target_qpos_dynamic_obs = target_qpos_dynamic_obs

        for st in range(self.substeps):
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:5] = alpha * target_qpos + (
                1. - alpha) * self._previous_target_qpos
            self.sim.data.ctrl[5:] = alpha * target_qpos_dynamic_obs + (
                1. - alpha) * self._previous_target_qpos_dynamic_obs
            self.sim.step()

        self.timestep += 1
        self._previous_target_qpos = target_qpos
        self._previous_target_qpos_dynamic_obs = target_qpos_dynamic_obs
        constraint = self.topple_check() or self.get_contact_info()
        reward = self.reward_fn()

        if EARLY_TERMINATION:
            done = (constraint > 0) or (reward > -0.5)
        else:
            done = False

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": position,
            "next_state": self.position,
            "action": action,
            "success": reward>-0.5
        }

        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info

    def get_contact_info(self):
        # Check for collision between arms
        contact_list = []
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            contact_list.append(contact.geom1)
            contact_list.append(contact.geom2)
        contact_list.sort()
        collision = False
        if 0 in contact_list or 1 in contact_list or 6 in contact_list or 7 in contact_list:
            collision = True
        return collision

    def topple_check(self, debug=False):
        quat = self.object_poses[:, 3:]
        phi = np.arctan2(
            2 *
            (np.multiply(quat[:, 0], quat[:, 1]) + quat[:, 2] * quat[:, 3]),
            1 - 2 * (np.power(quat[:, 1], 2) + np.power(quat[:, 2], 2)))
        theta = np.arcsin(2 * (np.multiply(quat[:, 0], quat[:, 2]) -
                               np.multiply(quat[:, 3], quat[:, 1])))
        psi = np.arctan2(
            2 * (np.multiply(quat[:, 0], quat[:, 3]) +
                 np.multiply(quat[:, 1], quat[:, 2])),
            1 - 2 * (np.power(quat[:, 2], 2) + np.power(quat[:, 3], 2)))
        euler = np.stack([phi, theta, psi]).T[:, :2] * 180. / np.pi
        if debug:
            return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0, euler
        return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0

    @property
    def jaw_width(self):
        pos = self.position
        return 0.08 - (pos[4] - pos[5])

    def set_y_range(self, bounds):
        self.obj_y_dist_range[0] = bounds[0]
        self.obj_y_dist_range[1] = bounds[1]

    def expert_action(self, t, noise_std=0.0, demo_quality='high'):
        cur_pos = self.position[:3]
        cur_pos[1] += 0.28  # compensate for length of jaws

        if t < 2:
            return np.clip([0, 0, 0.03, 0] +
                           np.random.randn(self._adim) * noise_std,
                           self.ac_low, self.ac_high)

        if demo_quality == 'high':
            thresh1 = 0.03
            thresh2 = 0.03
        else:
            thresh1 = 0.04
            thresh2 = 0.04

        target_obj_pos = self.object_poses[1][:3]
        action = np.zeros(self._adim)
        delta = target_obj_pos - cur_pos
        if np.abs(delta[0]) > thresh1:
            action[0] = 1.3 * delta[0]
            action[3] = 0.02
        elif np.abs(delta[1]) > thresh2:
            action[1] = 1.3 * delta[1]
            action[3] = 0.01
            if t > 6 and cur_pos[2] > 0.03:
                action[2] = -0.05
        elif self.jaw_width > 0.06:
            action[3] = 0.06
        else:
            action[3] = 0.06
            action[2] = 0.05
        return np.clip(action + np.random.randn(self._adim) * noise_std,
                       self.ac_low, self.ac_high)

    def reward_fn(self):
        return -(self.target_object_height <
                 self.target_height_thresh).astype(float)

    def object_reset_poses(self):
        new_poses = np.zeros((3, 7))
        new_poses[:, 3] = 1
        if self.randomize_objects == True:
            x = np.random.uniform(self.obj_x_range[0], self.obj_x_range[1])
            y1 = np.random.randn() * 0.05
            y0 = y1 - np.random.uniform(self.obj_y_dist_range[0],
                                        self.obj_y_dist_range[1])
            y2 = y1 + np.random.uniform(self.obj_y_dist_range[0],
                                        self.obj_y_dist_range[1])
            new_poses[0, 0:2] = np.array([y0, x])
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x])
        else:
            x = np.mean(self.obj_x_range)
            y1 = 0.
            y0 = y1 - np.mean(self.obj_y_dist_range)
            y2 = y1 + np.mean(self.obj_y_dist_range)
            new_poses[0, 0:2] = np.array([y0, x])
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x])
        return new_poses

    @property
    def position(self):
        return np.copy(self.sim.get_state().qpos[:])

    @property
    def object_poses(self):
        pos = self.position
        num_objs = (self.position.shape[0] - 12) // 7
        poses = []
        for i in range(num_objs):
            poses.append(np.copy(pos[i * 7 + 12:(i + 1) * 7 + 12]))
        return np.array(poses)

    @property
    def target_object_height(self):
        return self.object_poses[1, 2] - 0.072

    def _next_qpos(self, action):
        assert action.shape[0] == self._adim, action
        target = no_rot_dynamics(self._previous_target_qpos, action)
        target = clip_target_qpos(target, self.low_bound, self.high_bound)
        return target

    def _next_qpos_dynamic_obs(self, action):
        assert action.shape[0] == self._adim, action
        target = no_rot_dynamics(self._previous_target_qpos_dynamic_obs,
                                 action)
        target = clip_target_qpos(target, self.low_bound, self.high_bound)
        return target
