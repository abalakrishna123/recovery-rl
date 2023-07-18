import argparse
'''
Util to compile command line arguments for core script to run experiements
for Recovery RL (rrl_main.py)
'''


def get_args():
    # Global Parameters
    parser = argparse.ArgumentParser(description='Recovery RL Arguments')
    parser.add_argument('--env-name',
                        default='maze',
                        help='Gym environment (default: maze)')
    parser.add_argument('--logdir',
                        default='runs',
                        help='exterior log directory')
    parser.add_argument('--logdir_suffix',
                        default='',
                        help='log directory suffix')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='run on CUDA (default: False)')
    parser.add_argument('--cnn',
                        action='store_true',
                        help='visual observations (default: False)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--updates_per_step',
                        type=int,
                        default=1,
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps',
                        type=int,
                        default=100,
                        help='Steps sampling random actions (default: 100)')
    parser.add_argument(
        '--target_update_interval',
        type=int,
        default=1,
        help='Value target update per no. of updates per step (default: 1)')

    # Forward Policy (SAC)
    parser.add_argument(
        '--policy',
        default='Gaussian',
        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument(
        '--eval',
        type=bool,
        default=True,
        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning',
                        type=bool,
                        default=False,
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed',
                        type=int,
                        default=123456,
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps',
                        type=int,
                        default=1000000,
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_eps',
                        type=int,
                        default=1000000,
                        help='maximum number of episodes (default: 1000000)')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='hidden size (default: 256)')
    parser.add_argument('--replay_size',
                        type=int,
                        default=1000000,
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--task_demos',
                        action='store_true',
                        help='use task demos to pretrain safety critic')
    parser.add_argument('--num_task_transitions',
                        type=int,
                        default=10000000,
                        help='number of task transitions')
    parser.add_argument('--critic_pretraining_steps',
                        type=int,
                        default=3000,
                        help='gradient steps for critic pretraining')

    # Q risk
    parser.add_argument(
        '--pos_fraction',
        type=float,
        default=-1,
        help='fraction of positive examples for critic training')
    parser.add_argument('--gamma_safe',
                        type=float,
                        default=0.5,
                        help='discount factor for constraints (default: 0.9)')
    parser.add_argument('--eps_safe',
                        type=float,
                        default=0.1,
                        help='Qrisk threshold (default: 0.1)')
    parser.add_argument(
        '--tau_safe',
        type=float,
        default=0.0002,
        help='Qrisk target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument(
        '--safe_replay_size',
        type=int,
        default=1000000,
        help='size of replay buffer for Qrisk (default: 1000000)')
    parser.add_argument('--num_unsafe_transitions',
                        type=int,
                        default=10000,
                        help='number of unsafe transitions')
    parser.add_argument('--critic_safe_pretraining_steps',
                        type=int,
                        default=10000,
                        help='gradient steps for Qrisk pretraining')

    ################### Recovery RL ###################
    parser.add_argument('--use_recovery',
                        action='store_true',
                        help='use recovery policy')

    # Recovery RL MF Recovery
    parser.add_argument('--MF_recovery',
                        action='store_true',
                        help='model free recovery policy')
    parser.add_argument('--Q_sampling_recovery',
                        action='store_true',
                        help='sample actions over Qrisk for recovery')

    # Recovery RL MB Recovery (parameters for PETS)
    parser.add_argument(
        '-ca',
        '--ctrl_arg',
        action='append',
        nargs=2,
        default=[],
        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments'
    )
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        nargs=2,
        default=[],
        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides'
    )
    parser.add_argument(
        '--recovery_policy_update_freq',
        type=int,
        default=1,
        help='Model updated with new transitions every recovery_policy_update_freq episodes'
    )

    # Recovery RL Visual MB Recovery
    parser.add_argument(
        '--vismpc_recovery',
        action='store_true',
        help='use model-based visual planning for recovery policy')
    parser.add_argument('--load_vismpc',
                        action='store_true',
                        help='load pre-trained visual dynamics model')
    parser.add_argument('--model_fname',
                        default='image_maze_dynamics',
                        help='path to pre-trained visual dynamics model')
    parser.add_argument(
        '--beta',
        type=float,
        default=10,
        help='beta for training VAE for visual dynamics model (default: 10)')

    # Recovery RL Ablations
    parser.add_argument('--disable_offline_updates',
                        action='store_true',
                        help='only train Qrisk online')
    parser.add_argument('--disable_online_updates',
                        action='store_true',
                        help='only train Qrisk on offline data')
    parser.add_argument('--disable_action_relabeling',
                        action='store_true',
                        help='train task policy on recovery policy actions')
    parser.add_argument(
        '--add_both_transitions',
        action='store_true',
        help='use both task and recovery transitions to train task policy')

    ################### Comparisons ###################

    # Safety Editor
    parser.add_argument(
        '--use_safety_editor',
        type=bool,
        default=False,
        help='whether to use the safety editor')

    parser.add_argument(
        '--safety_editor_lambda',
        type=float,
        default=0,
        help='safety editor lambda parameter')

    # RP
    parser.add_argument(
        '--constraint_reward_penalty',
        type=float,
        default=0,
        help='reward penalty when a constraint is violated (default: 0)')

    # Lagrangian, RSPO
    parser.add_argument(
        '--DGD_constraints',
        action='store_true',
        help='use dual gradient descent to jointly optimize for task rewards + constraints'
    )
    parser.add_argument(
        '--use_constraint_sampling',
        action='store_true',
        help='sample actions with task policy and filter with Qrisk')
    parser.add_argument('--nu',
                        type=float,
                        default=0.01,
                        help='penalty term in Lagrangian objective')
    parser.add_argument('--update_nu',
                        action='store_true',
                        help='update Lagrangian penalty term')
    parser.add_argument(
        '--nu_schedule',
        action='store_true',
        help='use linear schedule for Lagrangian penalty term nu')
    parser.add_argument('--nu_start',
                        type=float,
                        default=1e3,
                        help='start value for nu (high)')
    parser.add_argument('--nu_end',
                        type=float,
                        default=0,
                        help='end value for nu (low)')

    # RCPO
    parser.add_argument('--RCPO', action='store_true', help='Use RCPO')
    parser.add_argument('--lambda_RCPO',
                        type=float,
                        default=0.01,
                        help='penalty term for RCPO (default: 0.01)')

    return parser.parse_args()
