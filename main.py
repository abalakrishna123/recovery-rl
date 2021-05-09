'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

# -*- coding: utf-8 -*-
import datetime
import gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import moviepy.editor as mpy
import cv2
from torch import nn, optim
from dotmap import DotMap

from recovery_rl.sac import SAC
from recovery_rl.replay_memory import ReplayMemory, ConstraintReplayMemory
from recovery_rl.MPC import MPC
from recovery_rl.VisualMPC import VisualMPC
from recovery_rl.model import VisualEncoderAttn, TransitionModel, VisualReconModel
from recovery_rl.utils import linear_schedule

from config import create_config
from arg_utils import get_args
from env.make_utils import register_env, make_env


TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
torchify = lambda x: torch.FloatTensor(x).to('cuda')

ENV_ID = {
    'navigation1': 'Navigation-v0',
    'navigation2': 'Navigation-v1',
    'maze': 'Maze-v0',
    'image_maze': 'ImageMaze-v0',
    'obj_extraction': 'ObjExtraction-v0',
    'obj_dynamic_extraction': 'ObjDynamicExtraction-v0',
}


def set_seed(seed, env):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)


def dump_logs(test_rollouts, train_rollouts, logdir):
    data = {"test_stats": test_rollouts, "train_stats": train_rollouts}
    with open(osp.join(logdir, "run_stats.pkl"), "wb") as f:
        pickle.dump(data, f)


def print_episode_info(rollout):
    num_violations = 0
    for inf in rollout:
        num_violations += int(inf['constraint'])
    print("final reward: %f" % rollout[-1]["reward"])
    if len(rollout[-1]["state"].shape) < 3:
        print(rollout[-1]["state"])
    print("num violations: %d" % num_violations)


def recovery_config_setup(args):
    ctrl_args = DotMap(**{key: val for (key, val) in args.ctrl_arg})
    cfg = create_config(args.env_name, "MPC", ctrl_args, args.override, logdir)
    cfg.pprint()
    return cfg


def experiment_setup(logdir, args):
    if args.use_recovery and not (
            args.ddpg_recovery or args.Q_sampling_recovery):
        register_env(args.env_name)
        cfg = recovery_config_setup(args)
        env = cfg.ctrl_cfg.env
        if not args.vismpc_recovery:
            recovery_policy = MPC(cfg.ctrl_cfg)
        else:
            encoder = VisualEncoderAttn(
                args.env_name, args.hidden_size, ch=3).to(device=TORCH_DEVICE)
            transition_model = TransitionModel(
                args.hidden_size,
                env.action_space.shape[0]).to(device=TORCH_DEVICE)
            residual_model = VisualReconModel(
                args.env_name, args.hidden_size).to(device=TORCH_DEVICE)

            dynamics_param_list = list(transition_model.parameters()) + list(
                residual_model.parameters()) + list(encoder.parameters())
            dynamics_optimizer = optim.Adam(
                dynamics_param_list, lr=3e-4, eps=1e-4)
            dynamics_finetune_optimizer = optim.Adam(
                transition_model.parameters(), lr=3e-4, eps=1e-4)

            if args.load_vismpc:
                if 'maze' in args.env_name:
                    model_dicts = torch.load(
                        os.path.join('models', args.model_fname,
                                     'model_19900.pth'))
                else:
                    model_dicts = torch.load(
                        os.path.join('models', args.model_fname,
                                     'model_199900.pth'))

                transition_model.load_state_dict(
                    model_dicts['transition_model'])
                residual_model.load_state_dict(model_dicts['residual_model'])
                encoder.load_state_dict(model_dicts['encoder'])
                dynamics_optimizer.load_state_dict(
                    model_dicts['dynamics_optimizer'])
            else:
                logdir = os.path.join('models', args.model_fname)
                os.makedirs(logdir, exist_ok=True)

            if args.vismpc_recovery:
                cfg.ctrl_cfg.encoder = encoder
                cfg.ctrl_cfg.transition_model = transition_model
                cfg.ctrl_cfg.residual_model = residual_model
                cfg.ctrl_cfg.dynamics_optimizer = dynamics_optimizer
                cfg.ctrl_cfg.dynamics_finetune_optimizer = dynamics_finetune_optimizer
                cfg.ctrl_cfg.hidden_size = args.hidden_size
                cfg.ctrl_cfg.beta = args.beta
                cfg.ctrl_cfg.logdir = logdir
                cfg.ctrl_cfg.batch_size = args.batch_size
                recovery_policy = VisualMPC(cfg.ctrl_cfg)
    else:
        register_env(args.env_name)
        recovery_policy = None
        env = make_env(args.env_name)
    set_seed(args.seed, env)
    agent = agent_setup(env, logdir, args)
    if args.use_recovery and not (
            args.ddpg_recovery or args.Q_sampling_recovery):
        recovery_policy.update_value_func(agent.safety_critic)
    return agent, recovery_policy, env


def agent_setup(env, logdir, args):
    agent = SAC(
        env.observation_space,
        env.action_space,
        args,
        logdir,
        tmp_env=make_env(args.env_name))
    return agent


def get_action(state, env, agent, recovery_policy, args, train=True):
    def recovery_thresh(state, action, agent, recovery_policy, args):
        if not args.use_recovery:
            return False

        critic_val = agent.safety_critic.get_value(
            torchify(state).unsqueeze(0),
            torchify(action).unsqueeze(0))

        if critic_val > args.eps_safe:
            return True
        return False

    policy_state = state
    if args.start_steps > total_numsteps and train:
        action = env.action_space.sample()  # Sample random action
    elif train:
        action = agent.select_action(policy_state)  # Sample action from policy
    else:
        action = agent.select_action(
            policy_state, eval=True)  # Sample action from policy

    if recovery_thresh(state, action, agent, recovery_policy, args):
        recovery = True
        if args.ddpg_recovery or args.Q_sampling_recovery:
            real_action = agent.safety_critic.select_action(state)
        else:
            real_action = recovery_policy.act(state, 0)
    else:
        recovery = False
        real_action = np.copy(action)
    return action, real_action, recovery


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def get_constraint_demos(env, args):
    # Get demonstrations
    task_demo_data = None
    obs_seqs = []
    ac_seqs = []
    constraint_seqs = []
    if not args.task_demos:
        if args.env_name == 'reacher':
            constraint_demo_data = pickle.load(
                open(
                    osp.join("demos", "dvrk_reach", "constraint_demos.pkl"),
                    "rb"))
            if args.cnn:
                constraint_demo_data = constraint_demo_data['images']
            else:
                constraint_demo_data = constraint_demo_data['lowdim']
        elif 'maze' in args.env_name:
            # Maze
            if args.env_name == 'maze':
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", args.env_name,
                                 "constraint_demos.pkl"), "rb"))
            else:
            # Image Maze
                demo_data = pickle.load(
                    open(osp.join("demos", args.env_name, "demos.pkl"), "rb"))
                constraint_demo_data = demo_data['constraint_demo_data']
                obs_seqs = demo_data['obs_seqs']
                ac_seqs = demo_data['ac_seqs']
                constraint_seqs = demo_data['constraint_seqs']
        elif 'extraction' in args.env_name:
            # Object Extraction, Object Extraction (Dynamic Obstacle)
            folder_name = args.env_name.split('_env')[0]
            constraint_demo_data = pickle.load(
                open(
                    osp.join("demos", folder_name, "constraint_demos.pkl"),
                    "rb"))
        else:
            # Navigation 1 and 2
            constraint_demo_data = env.transition_function(
                args.num_unsafe_transitions)
    else:
        if 'extraction' in args.env_name:
            folder_name = args.env_name.split('_env')[0]
            task_demo_data = pickle.load(
                open(
                    osp.join("demos", folder_name, "task_demos.pkl"),
                    "rb"))
            constraint_demo_data = pickle.load(
                open(
                    osp.join("demos", folder_name,
                             "constraint_demos.pkl"), "rb"))
            # Get all violations in front to get as many violations as possible
            constraint_demo_data_list_safe = []
            constraint_demo_data_list_viol = []
            for i in range(len(constraint_demo_data)):
                if constraint_demo_data[i][2] == 1:
                    constraint_demo_data_list_viol.append(
                        constraint_demo_data[i])
                else:
                    constraint_demo_data_list_safe.append(
                        constraint_demo_data[i])

            constraint_demo_data = constraint_demo_data_list_viol[:int(
                0.5 * args.num_unsafe_transitions
            )] + constraint_demo_data_list_safe
        else:
            constraint_demo_data, task_demo_data = env.transition_function(
                args.num_unsafe_transitions, task_demos=args.task_demos)
    return constraint_demo_data, task_demo_data, obs_seqs, ac_seqs, constraint_seqs

# Train MB recovery policy
def train_MB_recovery(states, actions, next_states=None, epochs=50):
    if next_states is not None:
        recovery_policy.train(
            states, actions, random=True, next_obs=next_states, epochs=epochs)
    else:
        recovery_policy.train(states, actions)

# Process observation for CNN
def process_obs(obs, env_name):
    if 'extraction' in args.env_name:
        obs = cv2.resize(obs, (64, 48), interpolation=cv2.INTER_AREA)
    im = np.transpose(obs, (2, 0, 1))
    return im


args = get_args()
# Logging setup
logdir = os.path.join(
    args.logdir, '{}_SAC_{}_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
        args.policy, args.logdir_suffix))
if not os.path.exists(logdir):
    os.makedirs(logdir)
print("LOGDIR: ", logdir)
pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb"))
# Experiment setup
agent, recovery_policy, env = experiment_setup(logdir, args)

# Memory
memory = ReplayMemory(args.replay_size)
recovery_memory = ConstraintReplayMemory(args.safe_replay_size)

total_numsteps = 0
updates = 0
# Get demos
task_demos = args.task_demos
constraint_demo_data, task_demo_data, obs_seqs, ac_seqs, constraint_seqs = get_constraint_demos(
    env, args)

# Get multiplier schedule for RSPO
if args.nu_schedule:
    nu_schedule = linear_schedule(args.nu_start, args.nu_end, args.num_eps)
else:
    nu_schedule = linear_schedule(args.nu, args.nu, 0)

# Train recovery policy and associated value function on demos
num_constraint_violations = 0
if not args.disable_offline_updates:
    if args.use_recovery or args.DGD_constraints or args.RCPO:
        if not args.vismpc_recovery:
            # Get data for recovery policy and safety critic training
            demo_data_states = np.array([
                d[0]
                for d in constraint_demo_data[:args.num_unsafe_transitions]
            ])
            demo_data_actions = np.array([
                d[1]
                for d in constraint_demo_data[:args.num_unsafe_transitions]
            ])
            demo_data_next_states = np.array([
                d[3]
                for d in constraint_demo_data[:args.num_unsafe_transitions]
            ])
            num_unsafe_transitions = 0
            for transition in constraint_demo_data:
                recovery_memory.push(*transition)
                num_constraint_violations += int(transition[2])
                num_unsafe_transitions += 1
                if num_unsafe_transitions == args.num_unsafe_transitions:
                    break
            print("Number of Constraint Transitions: ",
                  num_unsafe_transitions)
            print("Number of Constraint Violations: ",
                  num_constraint_violations)

            # Train DDPG recovery policy
            for i in range(args.critic_safe_pretraining_steps):
                if i % 100 == 0:
                    print("CRITIC SAFE UPDATE STEP: ", i)
                agent.safety_critic.update_parameters(
                    memory=recovery_memory,
                    policy=agent.policy,
                    critic=agent.critic,
                    batch_size=min(args.batch_size,
                                   len(constraint_demo_data)))

            # Train PETS recovery policy
            if not (args.ddpg_recovery or args.Q_sampling_recovery
                    or args.DGD_constraints or args.RCPO):
                train_MB_recovery(
                    demo_data_states,
                    demo_data_actions,
                    demo_data_next_states,
                    epochs=50)
        else:
            # Pre-train vis dynamics model if needed
            if not args.load_vismpc:
                recovery_policy.train(
                    obs_seqs,
                    ac_seqs,
                    constraint_seqs,
                    num_train_steps=20000
                    if "maze" in args.env_name else 200000)
            # Get data for recovery policy and safety critic training
            num_unsafe_transitions = 0
            for transition in constraint_demo_data:
                recovery_memory.push(*transition)
                num_constraint_violations += int(transition[2])
                num_unsafe_transitions += 1
                if num_unsafe_transitions == args.num_unsafe_transitions:
                    break
            print("Number of Constraint Transitions: ",
                  num_unsafe_transitions)
            print("Number of Constraint Violations: ",
                  num_constraint_violations)
            # Pass encoder to safety critic:
            agent.safety_critic.encoder = recovery_policy.get_encoding
            # Train safety critic on encoded states
            for i in range(args.critic_safe_pretraining_steps):
                if i % 100 == 0:
                    print("CRITIC SAFE UPDATE STEP: ", i)
                agent.safety_critic.update_parameters(
                    memory=recovery_memory,
                    policy=agent.policy,
                    critic=agent.critic,
                    batch_size=min(args.batch_size,
                                   len(constraint_demo_data)))

# Optionally initialize task policy with demos
if task_demos:
    # Get data for task policy
    num_task_transitions = 0
    for transition in task_demo_data:
        memory.push(*transition)
        num_task_transitions += 1
        if num_task_transitions == args.num_task_transitions:
            break
    print("Number of Task Transitions: ", num_task_transitions)
    # Pre-train task critic
    for i in range(args.critic_pretraining_steps):
        if i % 100 == 0:
            print("Update: ", i)
        agent.update_parameters(
            memory,
            min(args.batch_size, num_task_transitions),
            updates,
            safety_critic=agent.safety_critic)
        updates += 1

# Training Loop
test_rollouts = []
train_rollouts = []
all_ep_data = []

num_viols = 0
num_successes = 0
viol_and_recovery = 0
viol_and_no_recovery = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    if args.cnn:
        state = process_obs(state, args.env_name)

    train_rollouts.append([])
    ep_states = [state]
    ep_actions = []
    ep_constraints = []

    if i_episode % 10 == 0:
        print("SEED: ", args.seed)
        print("LOGDIR: ", logdir)

    while not done:
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory,
                    min(args.batch_size, len(memory)),
                    updates,
                    safety_critic=agent.safety_critic,
                    nu=nu_schedule(i_episode))
                if not args.disable_online_updates and len(
                        recovery_memory) > args.batch_size and (
                            num_viols + num_constraint_violations
                        ) / args.batch_size > args.pos_fraction:
                    agent.safety_critic.update_parameters(
                        memory=recovery_memory,
                        policy=agent.policy,
                        critic=agent.critic,
                        batch_size=args.batch_size,
                        plot=0)
                updates += 1

        # Get action, execute action, and compile step results
        action, real_action, recovery_used = get_action(
            state, env, agent, recovery_policy, args)
        next_state, reward, done, info = env.step(real_action)
        info['recovery'] = recovery_used

        if args.cnn:
            next_state = process_obs(next_state, args.env_name)

        train_rollouts[-1].append(info)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        if info['constraint']:
            reward -= args.constraint_reward_penalty

        mask = float(not done)
        done = done or episode_steps == env._max_episode_steps

        # Update buffers
        if not args.disable_action_relabeling:
            memory.push(state, action, reward, next_state,
                        mask)
        else:
            memory.push(state, real_action, reward, next_state,
                        mask)

        if args.use_recovery or args.DGD_constraints or args.RCPO:
            recovery_memory.push(state, real_action, info['constraint'],
                                 next_state, mask)
            if recovery_used and args.add_both_transitions:
                memory.push(state, real_action, reward, next_state,
                            mask)
        state = next_state
        ep_states.append(state)
        ep_actions.append(real_action)
        ep_constraints.append([info['constraint']])

    # Get success/violation stats
    if info['constraint']:
        num_viols += 1
        if info['recovery']:
            viol_and_recovery += 1
        else:
            viol_and_no_recovery += 1
    if "extraction" in args.env_name and info['reward'] > -0.5:
        num_successes += 1
    elif "navigation" in args.env_name and info['reward'] > -4:
        num_successes += 1
    elif "maze" in args.env_name and -info['reward'] < 0.03:
        num_successes += 1

    # Update recovery policy using online data
    if args.use_recovery and not args.disable_online_updates:
        all_ep_data.append({
            'obs': np.array(ep_states),
            'ac': np.array(ep_actions),
            'constraint': np.array(ep_constraints)
        })
        if i_episode % args.recovery_policy_update_freq == 0 and not (
                args.ddpg_recovery or args.Q_sampling_recovery
                or args.DGD_constraints):
            if not args.vismpc_recovery:
                train_MB_recovery([ep_data['obs'] for ep_data in all_ep_data],
                               [ep_data['ac'] for ep_data in all_ep_data])
                all_ep_data = []
            else:
                recovery_policy.train_online(
                    i_episode, recovery_memory
                )

    # Print performance stats
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".
          format(i_episode, total_numsteps, episode_steps,
                 round(episode_reward, 2)))
    print_episode_info(train_rollouts[-1])
    print("Num Violations So Far: %d" % num_viols)
    print("Violations with Recovery: %d" % viol_and_recovery)
    print("Violations with No Recovery: %d" % viol_and_no_recovery)
    print("Num Successes So Far: %d" % num_successes)

    if total_numsteps > args.num_steps or i_episode > args.num_eps:
        break

    # Get test rollouts
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 1
        for j in range(episodes):
            test_rollouts.append([])
            state = env.reset()

            if 'maze' in args.env_name:
                im_list = [env._get_obs(images=True)]
            elif 'extraction' in args.env_name:
                im_list = [env.render().squeeze()]

            if args.cnn:
                state = process_obs(state, args.env_name)

            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action, real_action, recovery_used = get_action(
                    state, env, agent, recovery_policy, args, train=False)
                next_state, reward, done, info = env.step(real_action)  # Step
                info['recovery'] = recovery_used
                done = done or episode_steps == env._max_episode_steps

                if 'maze' in args.env_name:
                    im_list.append(env._get_obs(images=True))
                elif 'extraction' in args.env_name:
                    im_list.append(env.render().squeeze())

                if args.cnn:
                    next_state = process_obs(next_state, args.env_name)

                test_rollouts[-1].append(info)
                episode_reward += reward
                episode_steps += 1
                state = next_state

            print_episode_info(test_rollouts[-1])
            avg_reward += episode_reward

            if 'maze' in args.env_name or 'extraction' in args.env_name:
                npy_to_gif(
                    im_list,
                    osp.join(logdir, "test_" + str(i_episode) + "_" + str(j)))

        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    dump_logs(test_rollouts, train_rollouts, logdir)

env.close()