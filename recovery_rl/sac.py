'''
Built on on SAC implementation from
https://github.com/pranz24/pytorch-soft-actor-critic
'''

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
    ValueNetwork, ValueNetworkCNN
from recovery_rl.qrisk import QRiskWrapper


class SAC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir,
                 im_shape=None,
                 tmp_env=None):

        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        # Parameters for SAC
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha # 0.5   entropy term

        self.conservative_safety_critics = args.conservative_safety_critic
        self.lam = args.lam # the lambda param to mix reward and risk
        self.eta = args.eta # 4e-2   the learning rate for the lambda
        self.trust_region = args.trust_region # 0.01   the policy trust region
        self.beta_init = args.beta_init # 0.7 the initial beta for the trust region optimization
        self.conservative_critic_update_l = args.conservative_critic_update_l # 20 the number of steps of line search
        self.env_name = args.env_name
        self.logdir = logdir
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.updates = 0
        self.cnn = args.cnn
        if im_shape:
            observation_space = im_shape

        # RRL specific parameters
        self.gamma_safe = args.gamma_safe
        self.eps_safe = args.eps_safe
        # Parameters for comparisonsp
        self.DGD_constraints = args.DGD_constraints
        self.nu = args.nu
        self.update_nu = args.update_nu

        self.use_constraint_sampling = args.use_constraint_sampling
        self.log_nu = torch.tensor(np.log(self.nu),
                                   requires_grad=True,
                                   device=self.device)
        self.nu_optim = Adam([self.log_nu], lr=0.1 * args.lr)

        self.RCPO = args.RCPO
        self.lambda_RCPO = args.lambda_RCPO
        self.log_lambda_RCPO = torch.tensor(np.log(self.lambda_RCPO),
                                            requires_grad=True,
                                            device=self.device)
        self.lambda_RCPO_optim = Adam(
            [self.log_lambda_RCPO],
            lr=0.1 * args.lr)  # Make lambda update slower for stability


        # SAC setup
        critic_cls = QNetworkCNN if args.cnn else QNetwork
        value_cls = ValueNetworkCNN if args.cnn else ValueNetwork

        print(observation_space)
        print(dir(observation_space))
        self.critic = critic_cls(observation_space.shape[0], action_space.shape[0],
                                 args.hidden_size, args.env_name).to(device=self.device)
        self.critic_target = critic_cls(observation_space.shape[0], action_space.shape[0],
                                        args.hidden_size, args.env_name).to(device=self.device)

        self.value_f = value_cls(observation_space.shape[0], args.hidden_size).to(device=self.device)
        self.value_f_target = value_cls(observation_space.shape[0], args.hidden_size).to(device=self.device)

        self.value_f_risk = value_cls(observation_space.shape[0], args.hidden_size).to(device=self.device)
        self.value_f_risk_target = value_cls(observation_space.shape[0], args.hidden_size).to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.value_optim = Adam(list(self.value_f.parameters()) + list(self.value_f_risk.parameters()), lr=args.lr)

        hard_update(self.critic_target, self.critic)
        hard_update(self.value_f_target, self.value_f)
        hard_update(self.value_f_risk_target, self.value_f_risk)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A)
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1,
                                             requires_grad=True,
                                             device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            if args.cnn:
                self.policy = GaussianPolicyCNN(observation_space,
                                                action_space.shape[0],
                                                args.hidden_size,
                                                args.env_name,
                                                action_space).to(self.device)
                if self.conservative_safety_critics:
                    self.inner_pi = GaussianPolicyCNN(
                        observation_space,
                        action_space.shape[0],
                        args.hidden_size,
                        args.env_name,
                        action_space).to(self.device)
            else:
                self.policy = GaussianPolicy(observation_space.shape[0],
                                             action_space.shape[0],
                                             args.hidden_size,
                                             action_space).to(self.device)

                if self.conservative_safety_critics:
                    self.inner_pi = GaussianPolicy(
                        observation_space.shape[0],
                        action_space.shape[0],
                        args.hidden_size,
                        action_space).to(self.device)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            assert not args.cnn
            self.policy = DeterministicPolicy(observation_space.shape[0],
                                              action_space.shape[0],
                                              args.hidden_size,
                                              action_space).to(self.device)
            if self.conservative_safety_critics:
                self.inner_pi = DeterministicPolicy(observation_space.shape[0],
                                                    action_space.shape[0],
                                                    args.hidden_size,
                                                    action_space).to(self.device)

        params = self.policy.parameters()
        if self.conservative_safety_critics:
            params = self.inner_pi.parameters()
            hard_update(self.inner_pi, self.policy)
        self.policy_optim = Adam(params, lr=args.lr)

        # Initialize safety critic
        self.safety_critic = QRiskWrapper(observation_space,
                                          action_space,
                                          args.hidden_size,
                                          logdir,
                                          args,
                                          tmp_env=tmp_env)

    def select_action(self, state, eval=False):
        '''
            Get action from current task policy
        '''
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # Action Sampling for SQRL
        if self.use_constraint_sampling:
            self.safe_samples = 100  # TODO: don't hardcode
            if not self.cnn:
                state_batch = state.repeat(self.safe_samples, 1)
            else:
                state_batch = state.repeat(self.safe_samples, 1, 1, 1)
            pi, log_pi, _ = self.policy.sample(state_batch)
            max_qf_constraint_pi = self.safety_critic.get_value(
                state_batch, pi)

            thresh_idxs = (max_qf_constraint_pi <= self.eps_safe).nonzero()[:,
                                                                            0]
            # Note: these are auto-normalized
            thresh_probs = torch.exp(log_pi[thresh_idxs])
            thresh_probs = thresh_probs.flatten()

            if list(thresh_probs.size())[0] == 0:
                min_q_value_idx = torch.argmin(max_qf_constraint_pi)
                action = pi[min_q_value_idx, :].unsqueeze(0)
            else:
                prob_dist = torch.distributions.Categorical(thresh_probs)
                sampled_idx = prob_dist.sample()
                action = pi[sampled_idx, :].unsqueeze(0)
        # Action Sampling for all other algorithms
        else:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self,
                          memory,
                          batch_size,
                          updates,
                          avg_violations,
                          nu=None,
                          safety_critic=None):
        '''
        Train task policy and associated Q function with experience in replay buffer (memory)
        '''
        if nu is None:
            nu = self.nu
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(
            self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_v_target = self.value_f_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * (
                next_state_v_target)
            if self.RCPO:
                qsafe_batch = torch.max(
                    *safety_critic(state_batch, action_batch))
                next_q_value -= self.lambda_RCPO * qsafe_batch

        v = self.value_f(state_batch)
        v_risk = self.value_f_risk(state_batch)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        sqf1_pi, sqf2_pi = self.safety_critic(state_batch, pi)
        max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)

        with torch.no_grad():
            v_target = min_qf_pi - self.alpha * log_pi
            v_risk_target = max_sqf_pi - self.alpha * log_pi
        v_loss = F.mse_loss(v, v_target)
        v_risk_loss = F.mse_loss(v_risk, v_risk_target)

        if self.DGD_constraints:
            policy_loss = (
                (self.alpha * log_pi) + nu * (max_sqf_pi - self.eps_safe) -
                1. * min_qf_pi
            ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        elif not self.conservative_safety_critics:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean(
            )  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        self.value_optim.zero_grad()
        (v_loss + v_risk_loss).backward()
        self.value_optim.step()

        if self.conservative_safety_critics:
            satisfied = False
            pi = self.inner_pi
            pi_params_save = pi.get_weights()

            beta = self.beta_init # 0.7

            for i in range(1, self.conservative_critic_update_l + 1): # 20

                # From A.2 the modified advantage function
                with torch.no_grad():
                    a_r = torch.min(*self.critic(state_batch, action_batch)) - v
                    a_c = torch.max(*self.safety_critic(state_batch, action_batch)) - v_risk
                    a_hat =  a_r - self.lam / (1. - self.gamma_safe) * a_c

                rnd = self.torchify(np.random.randn(*action_batch.shape))

                # Backtracking search for beta to satisfy the constraint
                # eq. 58 (A.2)
                candidate_params = []

                def inner_pi_loss_fn(*args):
                    p, log_p, _ = pi.sample_from_weights(state_batch, rnd, *args)
                    inner_pi_loss = (log_p * a_hat).mean()
                    return inner_pi_loss

                with torch.no_grad():
                    jaccs = torch.autograd.functional.jacobian(inner_pi_loss_fn, tuple(pi.parameters()))
                    info_mats = torch.autograd.functional.hessian(inner_pi_loss_fn, tuple(pi.parameters()))

                print(grads[0])
                print(jaccs[0])
                print(state_batch)
                print(len(grads), len(jaccs))
                print("DONE")
                exit()
                for grad, info_path, param in zip(jaccs, info_mats, pi.parameters()):
                    print(grad.shape, info_mat.shape, param.shape)
                    info_mat_inv = info_mat.pinverse()
                    print(grad.shape, info_mat.T.shape, grad.shape)
                    b = beta * torch.sqrt(
                        2. * self.trust_region / (grad @ info_mat.T @ grad + 1e-9)
                    )
                    print(param.shape, b.shape, info_mat_inv.shape, grad.shape)
                    candidate_params.append(param + (b @ info_mat_inv @ grad).view(*param.shape))

                pi.set_weights(candidate_params)
                with torch.no_grad():
                    kl = pi.kl_div(state_batch, *self.policy(state_batch)).mean()
                print("Kl div", kl)
                if kl <= self.trust_region:
                    print(kl, self.trust_region)
                    satisfied = True
                    print("Found update")
                    break

                # Revert params and decay the weight
                pi.set_weights(pi_params_save)
                beta = beta * (1. - beta)**i

            if satisfied:
                hard_update(self.policy, pi)

            hard_update(self.inner_pi, self.policy)























            policy_loss = v_loss

            # eq. 59 (A.2)
            self.lam = self.lam - self.eta * (
                1. / (1. - self.gamma) * a_c.mean() - (self.eps_safe - avg_violations)
            )
        else:
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()


        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # Optimize nu for LR
        if self.update_nu:
            nu_loss = (self.log_nu *
                       (self.eps_safe - max_sqf_pi).detach()).mean()
            self.nu_optim.zero_grad()
            nu_loss.backward()
            self.nu_optim.step()
            self.nu = self.log_nu.exp()

        # Optimize lambda for RCPO
        if self.RCPO:
            lambda_RCPO_loss = (self.log_lambda_RCPO *
                                (self.eps_safe - qsafe_batch).detach()).mean()
            self.lambda_RCPO_optim.zero_grad()
            lambda_RCPO_loss.backward()
            self.lambda_RCPO_optim.step()
            self.lambda_RCPO = self.log_lambda_RCPO.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.value_f_target, self.value_f, self.tau)
            soft_udpate(self.value_f_risk_target, self.value_f_risk, self.tau)

        return qf1_loss.item(), qf2_loss.item(), v_loss.item(), v_risk_loss.item(), \
            policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
