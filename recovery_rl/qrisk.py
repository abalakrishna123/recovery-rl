import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from dotmap import DotMap
import cv2

from recovery_rl.utils import soft_update, hard_update
from recovery_rl.model import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkCNN, \
    GaussianPolicyCNN, QNetworkConstraint, QNetworkConstraintCNN, DeterministicPolicyCNN, \
    StochasticPolicy, StochasticPolicyCNN


# Process observation for CNN
def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


'''
Wrapper for training, querying, and visualizing Q_risk for Recovery RL
'''


class QRiskWrapper:
    def __init__(self, obs_space, ac_space, hidden_size, logdir, action_space,
                 args, tmp_env):
        self.env_name = args.env_name
        self.logdir = logdir
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.ac_space = ac_space
        self.images = args.cnn
        self.encoding = args.vismpc_recovery
        if not self.images:
            self.safety_critic = QNetworkConstraint(
                obs_space.shape[0], ac_space.shape[0],
                hidden_size).to(device=self.device)
            self.safety_critic_target = QNetworkConstraint(
                obs_space.shape[0], ac_space.shape[0],
                args.hidden_size).to(device=self.device)
        else:
            if self.encoding:
                self.safety_critic = QNetworkConstraint(
                    hidden_size, ac_space.shape[0],
                    hidden_size).to(device=self.device)
                self.safety_critic_target = QNetworkConstraint(
                    hidden_size, ac_space.shape[0],
                    args.hidden_size).to(device=self.device)
            else:
                self.safety_critic = QNetworkConstraintCNN(
                    obs_space, ac_space.shape[0], hidden_size,
                    args.env_name).to(self.device)
                self.safety_critic_target = QNetworkConstraintCNN(
                    obs_space, ac_space.shape[0], hidden_size,
                    args.env_name).to(self.device)

        self.lr = args.lr
        self.safety_critic_optim = Adam(self.safety_critic.parameters(),
                                        lr=args.lr)
        hard_update(self.safety_critic_target, self.safety_critic)

        self.tau = args.tau_safe
        self.gamma_safe = args.gamma_safe
        self.updates = 0
        self.target_update_interval = args.target_update_interval
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        if not self.images:
            self.policy = StochasticPolicy(obs_space.shape[0],
                                              ac_space.shape[0], hidden_size,
                                              action_space).to(self.device)
        else:
            self.policy = StochasticPolicyCNN(obs_space, ac_space.shape[0],
                                                 hidden_size, args.env_name,
                                                 action_space).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.pos_fraction = args.pos_fraction if args.pos_fraction >= 0 else None
        self.ddpg_recovery = args.ddpg_recovery
        self.Q_sampling_recovery = args.Q_sampling_recovery
        self.tmp_env = tmp_env
        self.eps_safe = args.eps_safe
        self.alpha = args.alpha

        if args.env_name == 'maze':
            self.tmp_env.reset(pos=(12, 12))

    def update_parameters(self,
                          memory=None,
                          policy=None,
                          critic=None,
                          lr=None,
                          batch_size=None,
                          training_iterations=3000,
                          plot=1):
        '''
        Trains safety critic Q_risk and model-free recovery policy which performs
        gradient ascent on the safety critic

        Arguments:
            memory: Agent's replay buffer
            policy: Agent's composite policy
            critic: Safety critic (Q_risk)
        '''
        if self.pos_fraction:
            batch_size = min(batch_size,
                             int((1 - self.pos_fraction) * len(memory)))
        else:
            batch_size = min(batch_size, len(memory))
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        if self.encoding:
            state_batch_enc = self.encoder(state_batch)
            next_state_batch_enc = self.encoder(next_state_batch)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = policy.sample(
                next_state_batch)
            if self.encoding:
                qf1_next_target, qf2_next_target = self.safety_critic_target(
                    next_state_batch_enc, next_state_action)
            else:
                qf1_next_target, qf2_next_target = self.safety_critic_target(
                    next_state_batch, next_state_action)
            min_qf_next_target = torch.max(qf1_next_target, qf2_next_target)
            next_q_value = constraint_batch + mask_batch * self.gamma_safe * (
                min_qf_next_target)

        if self.encoding:
            qf1, qf2 = self.safety_critic(
                state_batch_enc, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
        else:
            qf1, qf2 = self.safety_critic(
                state_batch, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.safety_critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.safety_critic_optim.step()

        if self.ddpg_recovery:
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.safety_critic(state_batch, pi)
            max_sqf_pi = torch.max(qf1_pi, qf2_pi)
            policy_loss = max_sqf_pi.mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.safety_critic_target, self.safety_critic,
                        self.tau)
        self.updates += 1

        plot_interval = 1000
        if self.env_name == 'image_maze':
            plot_interval = 29000

        plot = False

        if plot and self.updates % 1000 == 0:
            if self.env_name in ['simplepointbot0', 'simplepointbot1', 'maze']:
                self.plot(policy, self.updates, [.1, 0], "right")
                self.plot(policy, self.updates, [-.1, 0], "left")
                self.plot(policy, self.updates, [0, .1], "up")
                self.plot(policy, self.updates, [0, -.1], "down")
            elif self.env_name == 'image_maze':
                self.plot(policy, self.updates, [.3, 0], "right")
                self.plot(policy, self.updates, [-.3, 0], "left")
                self.plot(policy, self.updates, [0, .3], "up")
                self.plot(policy, self.updates, [0, -.3], "down")
            else:
                raise NotImplementedError(
                    "Unsupported environment for plotting")

    def get_value(self, states, actions, encoded=False):
        '''
            Arguments:
                states, actions --> list of states and list of corresponding 
                actions to get Q_risk values for
            Returns: Q_risk(states, actions)
        '''
        with torch.no_grad():
            if self.encoding and not encoded:
                q1, q2 = self.safety_critic(self.encoder(states), actions)
            else:
                q1, q2 = self.safety_critic(states, actions)
            return torch.max(q1, q2)

    def select_action(self, state, eval=False):
        '''
            Gets action from model-free recovery policy

            Arguments:
                Current state
            Returns:
                action
        '''
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.ddpg_recovery:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
        elif self.Q_sampling_recovery:
            if not self.images:
                state_batch = state.repeat(1000, 1)
            else:
                state_batch = state.repeat(1000, 1, 1, 1)
            sampled_actions = torch.FloatTensor(
                np.array([self.ac_space.sample()
                          for _ in range(1000)])).to(self.device)
            q_vals = self.get_value(state_batch, sampled_actions)
            min_q_value_idx = torch.argmin(q_vals)
            action = sampled_actions[min_q_value_idx]
            return action.detach().cpu().numpy()
        else:
            assert False

    def plot(self, pi, ep, action=None, suffix="", critic=None):
        '''
            Interface for visualizing Q_risk for all navigation
            environments.
        '''
        env = self.tmp_env
        if self.env_name == 'maze':
            x_bounds = [-0.3, 0.3]
            y_bounds = [-0.3, 0.3]
        elif self.env_name == 'navigation1':
            x_bounds = [-80, 20]
            y_bounds = [-10, 10]
        elif self.env_name == 'navigation2':
            x_bounds = [-75, 25]
            y_bounds = [-75, 25]
        elif self.env_name == 'image_maze':
            x_bounds = [-0.05, 0.25]
            y_bounds = [-0.05, 0.25]
        else:
            raise NotImplementedError("Plotting unsupported for this env")

        states = []
        x_pts = 100
        y_pts = int(x_pts * (x_bounds[1] - x_bounds[0]) /
                    (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                if self.env_name == 'image_maze':
                    env.reset(pos=(x, y))
                    obs = process_obs(env._get_obs(images=True))
                    states.append(obs)
                else:
                    states.append([x, y])

        num_states = len(states)
        states = self.torchify(np.array(states))
        actions = self.torchify(np.tile(action, (len(states), 1)))

        if critic is None:
            if self.encoding:
                qf1, qf2 = self.safety_critic(self.encoder(states), actions)
            else:
                qf1, qf2 = self.safety_critic(states, actions)
            max_qf = torch.max(qf1, qf2)

        grid = max_qf.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)
        if self.env_name == 'navigation1':
            plt.gca().add_patch(
                Rectangle((0, 25),
                          500,
                          50,
                          linewidth=1,
                          edgecolor='r',
                          facecolor='none'))
        elif self.env_name == 'navigation2':
            plt.gca().add_patch(
                Rectangle((45, 65),
                          10,
                          20,
                          linewidth=1,
                          edgecolor='r',
                          facecolor='none'))

        if self.env_name == 'maze':
            background = cv2.resize(env._get_obs(images=True), (x_pts, y_pts))
            plt.imshow(background)
            plt.imshow(grid.T, alpha=0.6)
        else:
            plt.imshow(grid.T)

        plt.savefig(osp.join(self.logdir, "qvalue_" + str(ep) + suffix),
                    bbox_inches='tight')

    def __call__(self, states, actions):
        if self.encoding:
            return self.safety_critic(self.encoder(states), actions)
        else:
            return self.safety_critic(states, actions)
