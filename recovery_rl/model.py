'''
Latent dynamics models are built on latent dynamics model used in
Goal-Aware Prediction: Learning to Model What Matters (ICML 2020). All
other networks are built on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
'''
Global utilities
'''


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# Soft update of target critic network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


# Hard update of target critic network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


'''
Architectures for critic functions and policies for SAC model-free recovery
policies.
'''


# Q network architecture
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


# Q network architecture for image observations
class QNetworkCNN(nn.Module):
    def __init__(self, observation_space, num_actions, hidden_dim, env_name):
        super(QNetworkCNN, self).__init__()
        # Process the state
        self.conv1 = nn.Conv2d(observation_space[-1],
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(64,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)
        self.demo_bn1 = nn.BatchNorm2d(128)
        self.demo_bn2 = nn.BatchNorm2d(64)
        self.demo_bn3 = nn.BatchNorm2d(16)
        if 'shelf' in env_name:
            self.final_linear_size = 768
        elif 'maze' in env_name:
            self.final_linear_size = 1024
        elif "reach" in env_name:
            self.final_linear_size = 640
        else:
            assert (False, env_name)

        self.final_linear = nn.Linear(self.final_linear_size, hidden_dim)

        # Process the action
        self.linear_act1 = nn.Linear(num_actions, hidden_dim)
        self.linear_act2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_act3 = nn.Linear(hidden_dim, hidden_dim)

        # Q1 architecture

        # Post state-action merge
        self.linear1_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_1 = nn.Linear(hidden_dim, 1)

        # Post state-action merge
        self.linear1_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3_2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # Process the state
        bn1, bn2, bn3 = self.bn1, self.bn2, self.bn3

        conv1 = F.relu(bn1(self.conv1(state)))
        conv2 = F.relu(bn2(self.conv2(conv1)))
        conv3 = F.relu(bn3(self.conv3(conv2)))
        final_conv = conv3.view(-1, self.final_linear_size)

        final_conv = F.relu(self.final_linear(final_conv))

        # Process the action
        x0 = F.relu(self.linear_act1(action))
        x0 = F.relu(self.linear_act2(x0))
        x0 = self.linear_act3(x0)

        # Concat
        xu = torch.cat([final_conv, x0], 1)

        # Apply a few more FC layers in two branches
        x1 = F.relu(self.linear1_1(xu))
        x1 = F.relu(self.linear2_1(x1))

        x1 = self.linear3_1(x1)

        x2 = F.relu(self.linear1_2(xu))
        x2 = F.relu(self.linear2_2(x2))

        x2 = self.linear3_2(x2)
        return x1, x2


# Q_risk network architecture
class QNetworkConstraint(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkConstraint, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_inputs + num_actions)
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.sigmoid(self.linear3(x1))

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.sigmoid(self.linear6(x2))

        return x1, x2


# Q_risk network architecture for image observations
class QNetworkConstraintCNN(nn.Module):
    def __init__(self, observation_space, num_actions, hidden_dim, env_name):
        super(QNetworkConstraintCNN, self).__init__()
        # Process the state
        self.conv1 = nn.Conv2d(observation_space[-1],
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(64,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)
        self.demo_bn1 = nn.BatchNorm2d(128)
        self.demo_bn2 = nn.BatchNorm2d(64)
        self.demo_bn3 = nn.BatchNorm2d(16)
        if 'shelf' in env_name:
            self.final_linear_size = 768
        elif 'maze' in env_name:
            self.final_linear_size = 1024
        elif "reach" in env_name:
            self.final_linear_size = 640
        else:
            assert (False)

        self.final_linear = nn.Linear(self.final_linear_size, hidden_dim)

        # Process the action
        self.linear_act1 = nn.Linear(num_actions, hidden_dim)
        self.linear_act2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_act3 = nn.Linear(hidden_dim, hidden_dim)

        # Q1 architecture

        # Post state-action merge
        self.linear1_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_1 = nn.Linear(hidden_dim, 1)

        # Post state-action merge
        self.linear1_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3_2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # Process the state
        bn1, bn2, bn3 = self.bn1, self.bn2, self.bn3

        conv1 = F.relu(bn1(self.conv1(state)))
        conv2 = F.relu(bn2(self.conv2(conv1)))
        conv3 = F.relu(bn3(self.conv3(conv2)))
        final_conv = conv3.view(-1, self.final_linear_size)

        final_conv = F.relu(self.final_linear(final_conv))

        # Process the action
        x0 = F.relu(self.linear_act1(action))
        x0 = F.relu(self.linear_act2(x0))
        x0 = self.linear_act3(x0)

        # Concat
        xu = torch.cat([final_conv, x0], 1)

        # Apply a few more FC layers in two branches
        x1 = F.relu(self.linear1_1(xu))
        x1 = F.relu(self.linear2_1(x1))

        x1 = F.sigmoid(self.linear3_1(x1))

        x2 = F.relu(self.linear1_2(xu))
        x2 = F.relu(self.linear2_2(x2))

        x2 = F.sigmoid(self.linear3_2(x2))
        return x1, x2


# Gaussian policy for SAC
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


# Gaussian policy for SAC for image observations
class GaussianPolicyCNN(nn.Module):
    def __init__(self,
                 observation_space,
                 num_actions,
                 hidden_dim,
                 env_name,
                 action_space=None):
        super(GaussianPolicyCNN, self).__init__()
        # Process via a CNN and then collapse to linear
        self.conv1 = nn.Conv2d(observation_space[-1],
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(64,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)
        self.demo_bn1 = nn.BatchNorm2d(128)
        self.demo_bn2 = nn.BatchNorm2d(64)
        self.demo_bn3 = nn.BatchNorm2d(16)
        if 'shelf' in env_name:
            self.linear_dim = 768
        elif 'maze' in env_name:
            self.linear_dim = 1024
        elif "reach" in env_name:
            self.linear_dim = 640
        else:
            assert (False)

        self.linear1 = nn.Linear(self.linear_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        # Process the state
        bn1, bn2, bn3 = self.bn1, self.bn2, self.bn3

        conv1 = F.relu(bn1(self.conv1(state)))
        conv2 = F.relu(bn2(self.conv2(conv1)))
        conv3 = F.relu(bn3(self.conv3(conv2)))
        final_conv = conv3.view(-1, self.linear_dim)

        # Now do normal SAC stuff
        x = F.relu(self.linear1(final_conv))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyCNN, self).to(device)


# Deterministic policy for model free recovery
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


# Stochastic policy for model free recovery
class StochasticPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(StochasticPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = torch.nn.Parameter(
            torch.as_tensor([np.log(0.1)] * num_actions))
        self.min_log_std = np.log(1e-6)

        self.apply(weights_init_)
        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        #print(self.log_std)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        log_std = log_std.unsqueeze(0).repeat([len(mean), 1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(StochasticPolicy, self).to(device)


# Deterministic policy for model free recovery for image observations
class DeterministicPolicyCNN(nn.Module):
    def __init__(self,
                 observation_space,
                 num_actions,
                 hidden_dim,
                 env_name,
                 action_space=None):
        super(DeterministicPolicyCNN, self).__init__()
        # Process via a CNN and then collapse to linear
        self.conv1 = nn.Conv2d(observation_space[-1],
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(64,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)
        self.demo_bn1 = nn.BatchNorm2d(128)
        self.demo_bn2 = nn.BatchNorm2d(64)
        self.demo_bn3 = nn.BatchNorm2d(16)
        if 'shelf' in env_name:
            self.linear_dim = 768
        elif 'maze' in env_name:
            self.linear_dim = 1024
        elif "reach" in env_name:
            self.linear_dim = 640
        else:
            assert (False)

        self.linear1 = nn.Linear(self.linear_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        # Process the state
        bn1, bn2, bn3 = self.bn1, self.bn2, self.bn3

        conv1 = F.relu(bn1(self.conv1(state)))
        conv2 = F.relu(bn2(self.conv2(conv1)))
        conv3 = F.relu(bn3(self.conv3(conv2)))
        final_conv = conv3.view(-1, self.linear_dim)

        # Now do normal SAC stuff
        x = F.relu(self.linear1(final_conv))
        x = F.relu(self.linear2(x))

        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicyCNN, self).to(device)


# Stochastic policy for model free recovery for image observations
class StochasticPolicyCNN(nn.Module):
    def __init__(self,
                 observation_space,
                 num_actions,
                 hidden_dim,
                 env_name,
                 action_space=None):
        super(StochasticPolicyCNN, self).__init__()
        # Process via a CNN and then collapse to linear
        self.conv1 = nn.Conv2d(
            observation_space[-1],
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True)
        self.conv2 = nn.Conv2d(
            128, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(
            64, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)
        self.demo_bn1 = nn.BatchNorm2d(128)
        self.demo_bn2 = nn.BatchNorm2d(64)
        self.demo_bn3 = nn.BatchNorm2d(16)
        if 'shelf' in env_name:
            self.linear_dim = 768
        elif 'maze' in env_name:
            self.linear_dim = 1024
        elif "reach" in env_name:
            self.linear_dim = 640
        else:
            assert (False)

        self.linear1 = nn.Linear(self.linear_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = torch.nn.Parameter(
            torch.as_tensor([0.0] * num_actions))
        self.min_log_std = np.log(1e-6)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        # Process the state
        bn1, bn2, bn3 = self.bn1, self.bn2, self.bn3

        conv1 = F.relu(bn1(self.conv1(state)))
        conv2 = F.relu(bn2(self.conv2(conv1)))
        conv3 = F.relu(bn3(self.conv3(conv2)))
        final_conv = conv3.view(-1, self.linear_dim)

        # Now do normal SAC stuff
        x = F.relu(self.linear1(final_conv))
        x = F.relu(self.linear2(x))

        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        log_std = log_std.unsqueeze(0).repeat([len(mean), 1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(StochasticPolicyCNN, self).to(device)


'''
Architectures for latent dynamics model for model-based recovery policy
'''


# f_dyn, model of dynamics in latent space
class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, hidden_size, action_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(hidden_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, hidden_size)

    def forward(self, prev_hidden, action):
        hidden = torch.cat([prev_hidden, action], dim=-1)
        trajlen, batchsize = hidden.size(0), hidden.size(1)
        hidden.view(-1, hidden.size(2))
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.fc4(hidden)
        hidden = hidden.view(trajlen, batchsize, -1)
        return hidden


# Encoder
class VisualEncoderAttn(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 env_name,
                 hidden_size,
                 activation_function='relu',
                 ch=6):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.ch = ch
        self.conv1 = nn.Conv2d(self.ch, 32, 4, stride=2)  #3
        self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        if 'maze' in env_name:
            self.fc1 = nn.Linear(1024, 512)
        elif 'shelf' in env_name:
            self.fc1 = nn.Linear(512, 512)
        else:
            raise NotImplementedError("Needs to be maze or shelf")
        self.fc2 = nn.Linear(512, 2 * hidden_size)

    def forward(self, observation):
        trajlen, batchsize = observation.size(0), observation.size(1)
        self.width = observation.size(3)
        observation = observation.view(trajlen * batchsize, 3, self.width, 64)
        atn = torch.zeros_like(observation[:, :1])

        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv1_1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv2_1(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv3_1(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.act_fn(self.conv4_1(hidden))

        hidden = hidden.view(trajlen * batchsize, -1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.fc2(hidden)
        hidden = hidden.view(trajlen, batchsize, -1)
        atn = atn.view(trajlen, batchsize, 1, self.width, 64)
        return hidden, atn


# Decoder
class VisualReconModel(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 env_name,
                 hidden_size,
                 activation_function='relu',
                 action_len=5):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(hidden_size * 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.sigmoid = nn.Sigmoid()

        if 'maze' in env_name:
            self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
            self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
            self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
            self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        elif 'shelf' in env_name:
            self.conv1 = nn.ConvTranspose2d(128, 128, (4, 5), stride=2)
            self.conv2 = nn.ConvTranspose2d(128, 64, (4, 5), stride=2)
            self.conv3 = nn.ConvTranspose2d(64, 32, (5, 6), stride=2)
            self.conv4 = nn.ConvTranspose2d(32, 3, (4, 6), stride=2)
        else:
            raise NotImplementedError("Needs to be maze or shelf")

    def forward(self, hidden):
        trajlen, batchsize = hidden.size(0), hidden.size(1)
        hidden = hidden.view(trajlen * batchsize, -1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        hidden = hidden.view(-1, 128, 1, 1)

        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        residual = self.sigmoid(self.conv4(hidden)) * 255.0

        residual = residual.view(trajlen, batchsize, residual.size(1),
                                 residual.size(2), residual.size(3))
        return residual
