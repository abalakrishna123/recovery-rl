'''
Visual dynamics architecture is modelled on latent dynamics model used in
Goal-Aware Prediction: Learning to Model What Matters (ICML 2020)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from scipy.io import savemat
from tqdm import trange
import torch
from torchvision.utils import make_grid, save_image
import moviepy.editor as mpy
import matplotlib.pyplot as plt

from recovery_rl.utils import get_required_argument
from recovery_rl.optimizers import CEMOptimizer

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
torchify = lambda x: torch.FloatTensor(x).to(TORCH_DEVICE)


class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, get_pred_cost=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def process_obs(obs):
    im = np.transpose(obs, (2, 0, 1))
    return im


class VisualMPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .opt_cfg
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.

        """
        super().__init__(params)
        self.env = params.env
        self.temp_env = params.temp_env
        self.env_name = params.env_name
        self.dU = params.env.action_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.plan_hor = get_required_argument(
            params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.popsize = opt_cfg['popsize']
        self.num_elites = opt_cfg['num_elites']
        self.max_iters = opt_cfg['max_iters']
        self.alpha = opt_cfg['alpha']

        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(
            np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        self.encoder = params.encoder
        self.transition_model = params.transition_model
        self.residual_model = params.residual_model
        self.dynamics_optimizer = params.dynamics_optimizer
        self.dynamics_finetune_optimizer = params.dynamics_finetune_optimizer

        self.hidden_size = params.hidden_size
        self.beta = params.beta
        self.logdir = params.logdir
        self.batch_size = params.batch_size
        self.value_func = None

    def update_value_func(self, value_func):
        self.value_func = value_func

    def train_online(self, ep, memory, batch_size=3000, training_iterations=6):
        print("TRAIN DYNAMICS: ", ep)
        for j in range(training_iterations):
            state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(
                batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(TORCH_DEVICE)
            next_state_batch = torch.FloatTensor(next_state_batch).to(
                TORCH_DEVICE)
            state_enc_batch = self.encoder(
                state_batch.unsqueeze(0))[0].squeeze(0)[:, :self.hidden_size]
            next_state_enc_batch = self.encoder(next_state_batch.unsqueeze(
                0))[0].squeeze(0)[:, :self.hidden_size]
            action_batch = torch.FloatTensor(action_batch).to(TORCH_DEVICE)

            model_preds_enc_batch = self.transition_model(
                state_enc_batch.unsqueeze(0), action_batch.unsqueeze(0))
            model_preds_batch = self.residual_model(
                model_preds_enc_batch).squeeze(0)
            loss = ((model_preds_batch - next_state_batch)**2).mean()
            self.dynamics_finetune_optimizer.zero_grad()
            (loss).backward()
            self.dynamics_finetune_optimizer.step()

            if j % 5 == 0:
                with torch.no_grad():
                    print("Model Training Iteration %d    Loss: %f" %
                          (j, loss.detach().cpu().numpy()))

    def train(self,
              obs_seqs,
              ac_seqs,
              constraint_seqs,
              num_train_steps=20000,
              checkpoint_interval=100,
              curric_int=6):
        '''
        Offline visual dynamics training.

        Arguments:
            obs_seqs, ac_seqs, constraint_seqs: offline episodes of observations,
            actions, constraints used for visual dynamics model training
        '''
        metrics = {'trainsteps': [], 'observation_loss': [], 'teststeps': []}
        print("Number of Train Steps: ", num_train_steps)
        for s in range(num_train_steps):
            # Sample batch_size indices
            batch_idxs = np.random.randint(len(obs_seqs),
                                           size=self.batch_size).astype(int)
            obs_batch = torch.FloatTensor(obs_seqs[batch_idxs].transpose(
                1, 0, 2, 3, 4)).to(TORCH_DEVICE)
            action_batch = torch.FloatTensor(ac_seqs[batch_idxs].transpose(
                1, 0, 2)).to(TORCH_DEVICE)
            constraint_batch = torch.FloatTensor(
                constraint_seqs[batch_idxs].transpose(1, 0)).to(TORCH_DEVICE)
            # Get state encoding
            encoding, atn = self.encoder(obs_batch)
            mu, log_std = encoding[:, :, :self.
                                   hidden_size], encoding[:, :,
                                                          self.hidden_size:]
            std = torch.exp(log_std)
            samples = torch.empty(mu.shape).normal_(mean=0, std=1).cuda()
            encoding = mu + std * samples
            klloss = 0.5 * torch.mean(mu**2 + std**2 - torch.log(std**2) - 1)
            lossinc = min(curric_int - 1,
                          int(s / (num_train_steps / curric_int)))

            if s < num_train_steps:
                residuals = obs_batch
                all_losses = []
                # Pick random start frame for logging:
                sp_log = np.random.randint(obs_batch.size(0) - lossinc)
                for sp in range(obs_batch.size(0) - lossinc):
                    next_step = []
                    next_step_encoding = encoding[sp:sp + 1]
                    next_step.append(next_step_encoding)
                    for p in range(lossinc):
                        this_act = action_batch[sp + p:sp + p + 1]
                        next_step_encoding = self.transition_model(
                            next_step_encoding, this_act)
                        next_step.append(next_step_encoding)
                    next_step = torch.cat(next_step)
                    next_res = self.residual_model(next_step)
                    if sp == sp_log:
                        log_residual_pred = next_res
                    ## Reconstruction Error
                    prederr = ((residuals[sp:sp + 1 + lossinc] -
                                next_res[:1 + lossinc])**2)
                    all_losses.append(prederr.mean())
                r_loss = torch.stack(all_losses).mean(0)

            # Update all networks
            self.dynamics_optimizer.zero_grad()
            (r_loss + self.beta * klloss).backward()
            self.dynamics_optimizer.step()
            metrics['observation_loss'].append(r_loss.cpu().detach().numpy())
            metrics['trainsteps'].append(s)

            # Checkpoint models
            if s % checkpoint_interval == 0:
                print("Checkpoint: ", s)
                print("Loss Inc: ", lossinc)
                print("Observation Loss: ", r_loss.cpu().detach().numpy())
                print("KL Loss: ", klloss.cpu().detach().numpy())
                model_name = 'model_{}.pth'.format(s)

                torch.save(
                    {
                        'transition_model': self.transition_model.state_dict(),
                        'residual_model': self.residual_model.state_dict(),
                        'encoder': self.encoder.state_dict(),
                        'dynamics_optimizer':
                        self.dynamics_optimizer.state_dict(),
                    }, os.path.join(self.logdir, model_name))
                newpath = os.path.join(self.logdir, str(s))
                os.makedirs(newpath, exist_ok=True)
                metrics['teststeps'].append(s)
                # Save model predicttion gif
                video_frames = []
                for p in range(lossinc + 1):
                    video_frames.append(
                        make_grid(torch.cat([
                            residuals[p + sp_log, :5, :, :, :].cpu().detach(),
                            log_residual_pred[p, :5, :, :, :].cpu().detach(),
                        ],
                                            dim=3),
                                  nrow=1).numpy().transpose(1, 2, 0))

                npy_to_gif(video_frames,
                           os.path.join(newpath, 'train_steps_{}'.format(s)))

    def get_encoding(self, image):
        '''
        Get encoded version of image
        '''
        encoding, atn = self.encoder(image.unsqueeze(0))
        encoding = encoding[:, :, :self.hidden_size].squeeze(0)
        return encoding

    def act(self, obs, t, get_pred_cost=False):
        '''Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        '''

        # encode observation:
        encoding, atn = self.encoder(torchify(obs).unsqueeze(0).unsqueeze(0))
        encoding = encoding[:, :, :self.hidden_size]

        for itr in range(self.max_iters):
            if itr == 0:
                ## Generate random action samples for the first iteration
                action_samples = []
                for _ in range(self.popsize):
                    action_trajs = []
                    for j in range(self.plan_hor):
                        action_trajs.append(
                            torchify(self.env.action_space.sample()))
                    action_trajs = torch.stack(action_trajs)
                    action_samples.append(action_trajs)
                action_samples = torch.stack(action_samples).to(TORCH_DEVICE)
            else:
                sortid = costs.argsort()
                actions_sorted = action_samples[sortid]
                actions_ranked = actions_sorted[:self.num_elites]
                costs_ranked = costs[sortid][:self.num_elites]
                all_states_sorted = all_states[:, sortid, :]
                all_states_im_sorted = self.residual_model(all_states_sorted)
                all_states_im_ranked = all_states_im_sorted[:, :self.
                                                            num_elites, :, :, :]
                ## Refitting to Best Trajs
                mean, std = actions_ranked.mean(0), actions_ranked.std(0)
                smp = torch.empty(action_samples.shape).normal_(mean=0,
                                                                std=1).cuda()
                mean = mean.unsqueeze(0).repeat(self.popsize, 1, 1)
                std = std.unsqueeze(0).repeat(self.popsize, 1, 1)
                action_samples = smp * std + mean
                # TODO: Assuming action space is symmetric, should fix this eventually
                action_samples = torch.clamp(action_samples,
                                             min=self.env.action_space.low[0],
                                             max=self.env.action_space.high[0])

            curr_states = encoding.repeat(self.popsize, 1, 1)
            all_states = [curr_states]
            for j in range(self.plan_hor):
                next_states = self.transition_model(
                    curr_states, action_samples[:, j].unsqueeze(1))
                curr_states = next_states
                all_states.append(curr_states)

            all_states = torch.stack(all_states).squeeze()
            state_batch = all_states[:-1].transpose(1, 0)
            state_batch = state_batch.reshape(self.popsize * self.plan_hor, -1)
            action_batch = action_samples.reshape(self.popsize * self.plan_hor,
                                                  -1)
            costs = self.value_func.get_value(state_batch,
                                              action_batch,
                                              encoded=True)
            # Reshape back to normal
            costs = costs.reshape(self.popsize, self.plan_hor, -1)
            costs = torch.sum(
                costs, axis=1).squeeze()  # costs of all action sequences

        # Return the best action
        action = actions_ranked[0][0]
        return action.detach().cpu().numpy()
