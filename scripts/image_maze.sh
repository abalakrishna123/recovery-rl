#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..10}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name image_maze --use_recovery --MF_recovery --gamma_safe 0.65 --eps_safe 0.1 --cnn  --critic_safe_pretraining_steps 30000 --num_unsafe_transitions 20000 --logdir image_maze --logdir_suffix RRL_MF --num_eps 500 --seed $i
done

# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name image_maze --use_recovery --recovery_policy_update_freq 200000 --gamma_safe 0.6 --eps_safe 0.05 --cnn --critic_safe_pretraining_steps 30000 --num_unsafe_transitions 20000 --model_fname image_maze_dynamics --beta 10 --vismpc_recovery --load_vismpc --logdir image_maze --logdir_suffix RRL_MB --num_eps 500 --seed $i
done

# Unconstrained
for i in {1..10}
do
	echo "Unconstrained Run $i"
	python -m rrl_main --env-name image_maze --cuda --cnn --logdir image_maze --logdir_suffix unconstrained --num_eps 500 --seed $i
done

# Lagrangian Relaxation
for i in {1..10}
do
	echo "LR Run $i"
	python -m rrl_main --cuda --env-name image_maze --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu 10  --critic_safe_pretraining_steps 30000 --num_unsafe_transitions 20000 --update_nu --logdir image_maze --logdir_suffix LR --num_eps 500 --seed $i
done

# RSPO
for i in {1..10}
do
	echo "RSPO Run $i"
	python -m rrl_main --cuda --env-name image_maze --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --nu_schedule --nu_start 20  --critic_safe_pretraining_steps 30000 --num_unsafe_transitions 20000 --logdir image_maze --logdir_suffix RSPO --num_eps 500 --seed $i
done

# SQRL
for i in {1..10}
do
    echo "SQRL Run $i"
    python -m rrl_main --cuda --env-name image_maze --gamma_safe 0.65 --eps_safe 0.1 --cnn --DGD_constraints --use_constraint_sampling --nu 10 --update_nu  --critic_safe_pretraining_steps 30000 --num_unsafe_transitions 20000 --logdir image_maze --logdir_suffix SQRL --num_eps 500 --seed $i
done

# Reward Penalty
for i in {1..10}
do
	echo "RP Run $i"
	python -m rrl_main --env-name image_maze --cuda --constraint_reward_penalty 20 --cnn --logdir image_maze --logdir_suffix RP --num_eps 500 --seed $i
done

# RCPO
for i in {1..10}
do
	echo "RCPO Run $i"
	python -m rrl_main --cuda --env-name image_maze --gamma_safe 0.65 --eps_safe 0.1 --cnn --RCPO --lambda 20  --critic_safe_pretraining_steps 30000 --num_unsafe_transitions 20000 --logdir image_maze --logdir_suffix RCPO --num_eps 500 --seed $i
done