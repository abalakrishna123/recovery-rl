import os
import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from plotting_utils import get_color, get_legend_name

# ** Set plot type here **
PLOT_TYPE = "ratio"
assert PLOT_TYPE in ['ratio', 'success', 'violation', 'PR', 'reward']


def get_stats(data):
    minlen = min([len(d) for d in data])
    data = [d[:minlen] for d in data]
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0) / np.sqrt(len(data))
    ub = mu + np.std(data, axis=0) / np.sqrt(len(data))
    return mu, lb, ub


def moving_average(x, N):
    window_means = []
    for i in range(len(x) - N + 1):
        window = x[i:i + N]
        num_nans = np.count_nonzero(np.isnan(window))
        window_sum = np.nansum(window)
        if num_nans < N:
            window_mean = window_sum / (N - num_nans)
        else:
            window_mean = np.nan
        window_means.append(window_mean)
    return window_means


eps = {
    "navigation1": 300,
    "navigation2": 300,
    "maze": 500,
    "image_maze": 500,
    "obj_extraction": 4000,
    "object_dynamic_extraction": 3000,
    "method_ablations": 4000,
    "demo_ablations": 4000,
    "rp_ablations": 4000,
    "lr_ablations": 4000,
    "rcpo_ablations": 4000,
    "recovery_gamma_0.85_ablations": 4000,
    "recovery_gamma_0.75_ablations": 4000,
    "recovery_gamma_0.95_ablations": 4000,
}

envname = {
    "navigation1": "Navigation 1",
    "navigation2": "Navigation 2",
    "maze": "Maze",
    "image_maze": "Image Maze",
    "obj_extraction": "Object Extraction",
    "object_dynamic_extraction": "Object Extraction (Dynamic Obstacle)",
    "method_ablations": "Object Extraction: # Offline Transitions",
    "demo_ablations": "Object Extraction: Method Ablations",
    "rp_ablations": "Reward Penalty Ablation",
    "lr_ablations": "Lagrangian Relaxation Ablation",
    "rcpo_ablations": "RCPO Ablation",
    "recovery_gamma_0.85_ablations": "Recovery RL ($\gamma = 0.85$) Ablation",
    "recovery_gamma_0.75_ablations": "Recovery RL ($\gamma = 0.75$) Ablation",
    "recovery_gamma_0.95_ablations": "Recovery RL ($\gamma = 0.95$) Ablation",
}

if PLOT_TYPE == "ratio":
    yscaling = {
        "navigation1": 0.3,  # TODO: @ Brijen put in real value
        "navigation2": 0.3,  # TODO: @ Brijen put in real value
        "maze": 0.35,
        "image_maze": 0.55,
        "obj_extraction": 0.06,
        "object_dynamic_extraction": 0.06,  # TODO: @ Brijen put in real value
        "method_ablations": 0.06,
        "demo_ablations": 0.06,
        "rp_ablations": 0.04,
        "lr_ablations": 0.04,
        "rcpo_ablations": 0.04,
        "recovery_gamma_0.95_ablations": 0.04,
        "recovery_gamma_0.85_ablations": 0.04,
        "recovery_gamma_0.75_ablations": 0.15
    }
elif PLOT_TYPE == "success":
    yscaling = {
        "navigation1": 1,  # TODO: @ Brijen put in real value
        "navigation2": 1,  # TODO: @ Brijen put in real value
        "maze": 1,
        "image_maze": 1,
        "obj_extraction": 0.45,
        "object_dynamic_extraction": 0.45,  # TODO: @ Brijen put in real value
        "method_ablations": 0.0,
        "demo_ablations": 0.6,
        "rp_ablations": 0.6,
        "lr_ablations": 0.6,
        "rcpo_ablations": 0.6,
        "recovery_gamma_0.95_ablations": 0.6,
        "recovery_gamma_0.85_ablations": 0.6,
        "recovery_gamma_0.75_ablations": 0.6
    }
elif PLOT_TYPE == "violation":
    yscaling = {
        "navigation1": 0.3,  # TODO: @ Brijen put in real value
        "navigation2": 0.3,  # TODO: @ Brijen put in real value
        "maze": 0.25,
        "image_maze": 0.15,
        "obj_extraction": 0.07,
        "object_dynamic_extraction": 0.07,  # TODO: @ Brijen put in real value
        "method_ablations": 0.07,
        "demo_ablations": 0.07,
        "rp_ablations": 0.07,
        "lr_ablations": 0.07,
        "rcpo_ablations": 0.07,
        "recovery_gamma_0.95_ablations": 0.07,
        "recovery_gamma_0.85_ablations": 0.07,
        "recovery_gamma_0.75_ablations": 0.07
    }
else:
    yscaling = {
        "navigation1": 1,
        "navigation2": 1,
        "maze": 1,
        "image_maze": 1,
        "obj_extraction": 1,
        "object_dynamic_extraction": 1,
        "method_ablations": 1,
        "demo_ablations": 1,
        "rp_ablations": 1,
        "lr_ablations": 1,
        "rcpo_ablations": 1,
        "recovery_gamma_0.95_ablations": 1,
        "recovery_gamma_0.85_ablations": 1,
        "recovery_gamma_0.75_ablations": 1
    }


def plot_experiment(experiment, logdir):
    '''
        Construct experiment map for this experiment
    '''
    experiment_map = {}
    experiment_map['algs'] = {}
    for fname in os.listdir(logdir):
        if fname == '2021-04-08_16-59-36_SAC_obj_extraction_Gaussian_RRL_MF':
            continue
        alg_name = fname.split('Gaussian_')[-1]
        if alg_name not in experiment_map['algs']:
            experiment_map["algs"][alg_name] = [fname]
        else:
            experiment_map["algs"][alg_name].append(fname)

    experiment_map["outfile"] = osp.join('plotting', experiment + ".png")
    '''
        Save plot for experiment
    '''
    print("EXP NAME: ", experiment)
    max_eps = eps[experiment]
    fig, axs = plt.subplots(1, figsize=(16, 8))

    axs.set_title(envname[experiment], fontsize=48)
    axs.set_ylim(-0.1, int(yscaling[experiment] * max_eps) + 1)
    axs.set_xlabel("Episode", fontsize=42)
    if PLOT_TYPE == 'ratio':
        axs.set_ylabel("Ratio of Successes/Violations", fontsize=42)
    elif PLOT_TYPE == 'success':
        axs.set_ylabel("Cumulative Task Successes", fontsize=42)
    elif PLOT_TYPE == 'violation':
        axs.set_ylabel("Cumulative Constraint Violations", fontsize=42)
    elif PLOT_TYPE == 'PR':
        axs.set_ylabel("Task Successes", fontsize=42)
        axs.set_xlabel("Constraint Violations", fontsize=42)
    elif PLOT_TYPE == 'reward':
        axs.set_ylabel("Reward", fontsize=42)
    else:
        raise NotImplementedError("Unsupported Plot Type")

    axs.tick_params(axis='both', which='major', labelsize=36)
    plt.subplots_adjust(hspace=0.3)
    final_ratios_dict = {}

    for alg in experiment_map["algs"]:
        print(alg)
        exp_dirs = experiment_map["algs"][alg]
        fnames = [osp.join(exp_dir, "run_stats.pkl") for exp_dir in exp_dirs]

        task_successes_list = []
        train_rewards_safe_list = []
        train_violations_list = []

        for fname in fnames:
            with open(osp.join(logdir, fname), "rb") as f:
                data = pickle.load(f)
            train_stats = data['train_stats']

            train_violations = []
            train_rewards = []
            last_rewards = []

            for traj_stats in train_stats:
                train_violations.append([])
                train_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    train_violations[-1].append(step_stats['constraint'])
                    train_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']

                last_rewards.append(last_reward)

            ep_lengths = np.array([len(t) for t in train_violations])[:max_eps]
            train_violations = np.array(
                [np.sum(t) > 0 for t in train_violations])[:max_eps]

            train_violations = np.cumsum(train_violations)

            train_rewards = np.array(train_rewards)[:max_eps]
            train_rewards_safe = train_rewards
            train_rewards_safe[train_violations > 0] = np.nan

            last_rewards = np.array(last_rewards)[:max_eps]

            if 'maze' in experiment:
                task_successes = (-last_rewards < 0.03).astype(int)
            elif 'extraction' in experiment:
                task_successes = (last_rewards == 0).astype(int)
            elif "navigation" in experiment:
                task_successes = (last_rewards > -4).astype(int)
            else:
                task_successes = (last_rewards > -4).astype(int)

            task_successes = np.cumsum(task_successes)
            task_successes_list.append(task_successes)
            train_rewards_safe_list.append(train_rewards_safe)
            train_violations_list.append(train_violations)

        task_successes_list = np.array(task_successes_list)
        train_violations_list = np.array(train_violations_list)

        # Smooth out train rewards
        for i in range(len(train_rewards_safe_list)):
            train_rewards_safe_list[i] = moving_average(
                train_rewards_safe_list[i], 100)

        train_rewards_safe_list = np.array(train_rewards_safe_list)

        safe_ratios = (task_successes_list + 1) / (train_violations_list + 1)
        final_ratio = safe_ratios.mean(axis=0)[-1]
        final_successes = task_successes_list[:, -1]
        final_violations = train_violations_list[:, -1]

        final_success_mean = np.mean(final_successes)
        final_success_err = np.std(final_successes) / np.sqrt(
            len(final_successes))
        final_violation_mean = np.mean(final_violations)
        final_violation_err = np.std(final_violations) / np.sqrt(
            len(final_violations))

        final_ratios_dict[alg] = final_ratio
        safe_ratios_mean, safe_ratios_lb, safe_ratios_ub = get_stats(
            safe_ratios)
        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)
        trew_mean, trew_lb, trew_ub = get_stats(train_rewards_safe_list)

        if PLOT_TYPE == 'ratio':
            axs.fill_between(range(safe_ratios_mean.shape[0]),
                             safe_ratios_ub,
                             safe_ratios_lb,
                             color=get_color(alg),
                             alpha=.25,
                             label=get_legend_name(alg))
            axs.plot(safe_ratios_mean, color=get_color(alg))
        elif PLOT_TYPE == 'success':
            axs.fill_between(range(ts_mean.shape[0]),
                             ts_ub,
                             ts_lb,
                             color=get_color(alg),
                             alpha=.25,
                             label=get_legend_name(alg))
            axs.plot(ts_mean, color=get_color(alg))
        elif PLOT_TYPE == 'violation':
            axs.fill_between(range(tv_mean.shape[0]),
                             tv_ub,
                             tv_lb,
                             color=get_color(alg),
                             alpha=.25,
                             label=get_legend_name(alg))
            axs.plot(tv_mean, color=get_color(alg))
        elif PLOT_TYPE == 'PR':
            axs.errorbar([final_violation_mean], [final_success_mean],
                         xerr=[final_violation_err],
                         yerr=[final_success_err],
                         fmt='-o',
                         markersize=20,
                         linewidth=5,
                         color=get_color(alg),
                         label=get_legend_name(alg))
        elif PLOT_TYPE == 'reward':
            axs.fill_between(range(trew_mean.shape[0]),
                             trew_ub,
                             trew_lb,
                             color=get_color(alg),
                             alpha=.25,
                             label=get_legend_name(alg))
            axs.plot(trew_mean, color=get_color(alg))
        else:
            raise NotImplementedError("Unsupported Plot Type")

    print("FINAL RATIOS: ", final_ratios_dict)
    axs.legend(loc="upper left", fontsize=36, frameon=False)
    plt.savefig(experiment_map["outfile"], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    experiment = "obj_extraction"  # ** insert experiment name here **
    logdir = '/data/recovery-rl/obj_extraction'  # ** insert logdir here **
    plot_experiment(experiment, logdir)
