import pickle
import numpy as np

with open('runs/2021-04-25_06-07-00_SAC_maze_Gaussian_/run_stats.pkl', "rb") as f:
    data = pickle.load(f)
train_stats = data['train_stats']

train_violations = []
last_rewards = []
for traj_stats in train_stats:
	train_violations.append([])
	last_reward = 0
	for step_stats in traj_stats:
		train_violations[-1].append(step_stats['constraint'])
		last_reward = step_stats['reward']
	last_rewards.append(last_reward)
last_rewards = np.array(last_rewards)
print(last_rewards)
task_successes = np.sum((-last_rewards < 0.03).astype(int))
train_violations = np.array([np.sum(t) > 0 for t in train_violations])
train_violations = np.sum(train_violations)
print("SUCCESS: ", task_successes)
print("VIOLS: ", train_violations)
