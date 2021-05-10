import os
import pickle
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim
import cv2
"""
Constants associated with the Maze env.
"""

HORIZON = 100
MAX_FORCE = 0.1
FAILURE_COST = 0
GOAL_THRESH = 3e-2

GT_STATE = True
DENSE_REWARD = True


def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)


def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


def get_offline_data(num_transitions, images=False, save_rollouts=False):
    env = MazeNavigation()
    transitions = []
    num_constraints = 0
    total = 0
    rollouts = []

    for i in range(1 * num_transitions // 2):
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.3:
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False)
            rollouts.append([])

            if images:
                im_state = env.sim.render(64, 64, camera_name="cam0")
                im_state = process_obs(im_state)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if images:
            im_next_state = env.sim.render(64, 64, camera_name="cam0")
            im_next_state = process_obs(im_next_state)
        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, not done))
        transitions.append((state, action, constraint, next_state, not done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        if images:
            im_state = im_next_state

    for i in range(1 * num_transitions // 2):
        if i % 20 == 0:
            sample = np.random.uniform(0, 1, 1)[0]
            if sample < 0.3:
                mode = 'e'
            elif sample < 0.6:
                mode = 'm'
            else:
                mode = 'h'
            state = env.reset(mode, check_constraint=False)
            rollouts.append([])

            if images:
                im_state = env.sim.render(64, 64, camera_name="cam0")
                im_state = process_obs(im_state)
        action = env.expert_action()
        next_state, reward, done, info = env.step(action)
        if images:
            im_next_state = env.sim.render(64, 64, camera_name="cam0")
            im_next_state = process_obs(im_next_state)
        constraint = info['constraint']

        rollouts[-1].append((state, action, constraint, next_state, not done))
        transitions.append((state, action, constraint, next_state, not done))

        total += 1
        num_constraints += int(constraint)
        state = next_state
        if images:
            im_state = im_next_state

    print("data dist", total, num_constraints)
    if save_rollouts:
        return rollouts
    else:
        return transitions


class MazeNavigation(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'assets/simple_maze.xml')
        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = HORIZON
        self._max_episode_steps = self.horizon
        self.transition_function = get_offline_data
        self.steps = 0
        self.images = not GT_STATE
        self.action_space = Box(-MAX_FORCE * np.ones(2),
                                MAX_FORCE * np.ones(2))
        self.transition_function = get_offline_data
        obs = self._get_obs()
        self.dense_reward = DENSE_REWARD

        if self.images:
            self.observation_space = obs.shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=obs.shape)

        self.gain = 1.05
        self.goal = np.zeros((2, ))
        self.goal[0] = 0.25
        self.goal[1] = 0

    def step(self, action):
        action = process_action(action)
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = action
        cur_obs = self._get_obs()
        constraint = int(self.sim.data.ncon > 3)
        if not constraint:
            for _ in range(500):
                self.sim.step()
        obs = self._get_obs()
        self.sim.data.qvel[:] = 0
        self.steps += 1
        constraint = int(self.sim.data.ncon > 3)
        self.done = self.steps >= self.horizon or constraint or (
            self.get_distance_score() < GOAL_THRESH)
        if not self.dense_reward:
            reward = -(self.get_distance_score() > GOAL_THRESH).astype(float)
        else:
            reward = -self.get_distance_score()

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": cur_obs,
            "next_state": obs,
            "action": action,
            "success": reward>-0.03
        }

        return obs, reward, self.done, info

    def _get_obs(self, images=False):
        if images:
            return self.sim.render(64, 64, camera_name="cam0")
        # Joint poisitions and velocities
        state = np.concatenate(
            [self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])

        if not self.images and not images:
            return state[:2]  # State is just (x, y) position

        # Get images
        ims = self.sim.render(64, 64, camera_name="cam0")
        return ims / 255

    def reset(self, difficulty='h', check_constraint=True, pos=()):
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty is None:
                self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
            elif difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.14, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            elif difficulty == 'h':
                self.sim.data.qpos[0] = np.random.uniform(-0.22, -0.13)
            self.sim.data.qpos[1] = np.random.uniform(-0.22, 0.22)

        self.steps = 0

        w1 = -0.08
        w2 = 0.08
        self.sim.model.geom_pos[5, 1] = 0.5 + w1
        self.sim.model.geom_pos[7, 1] = -0.25 + w1
        self.sim.model.geom_pos[6, 1] = 0.4 + w2
        self.sim.model.geom_pos[8, 1] = -0.25 + w2
        self.sim.forward()
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty)

        return self._get_obs()

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
            """
        d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
        return d

    def expert_action(self):
        st = self.sim.data.qpos[:]
        if st[0] <= -0.151:
            delt = (np.array([-0.15, -0.125]) - st)
        elif st[0] <= 0.149:
            delt = (np.array([0.15, 0.125]) - st)
        else:
            delt = (np.array([self.goal[0], self.goal[1]]) - st)
        act = self.gain * delt

        return act
