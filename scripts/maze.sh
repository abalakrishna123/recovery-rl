#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..10}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name maze --use_recovery --MF_recovery --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --logdir maze --logdir_suffix RRL_MF --num_eps 500 --seed $i
done

# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name maze --use_recovery --recovery_policy_update_freq 5 --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --logdir maze --logdir_suffix RRL_MB --num_eps 500 --seed $i
done

# Unconstrained
for i in {1..10}
do
	echo "Unconstrained Run $i"
	python -m rrl_main --env-name maze --cuda --logdir maze --logdir_suffix unconstrained --num_eps 500 --seed $i
done

# Lagrangian Relaxation
for i in {1..10}
do
	echo "LR Run $i"
	python -m rrl_main --cuda --env-name maze --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu 100 --update_nu --logdir maze --logdir_suffix LR --num_eps 500 --seed $i
done

# RSPO
for i in {1..10}
do
	echo "RSPO Run $i"
	python -m rrl_main --cuda --env-name maze --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --nu_schedule --nu_start 200 --logdir maze --logdir_suffix RSPO --num_eps 500 --seed $i
done

# SQRL
for i in {1..10}
do
	echo "SQRL Run $i"
	python -m rrl_main --cuda --env-name maze --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --DGD_constraints --use_constraint_sampling --nu 100 --update_nu --logdir maze --logdir_suffix SQRL --num_eps 500 --seed $i
done

# Reward Penalty
for i in {1..10}
do
	echo "RP Run $i"
	python -m rrl_main --env-name maze --cuda --constraint_reward_penalty 50  --logdir maze --logdir_suffix RP --num_eps 500 --seed $i
done

# RCPO
for i in {1..10}
do
	echo "RCPO Run $i"
	python -m rrl_main --cuda --env-name maze --gamma_safe 0.5 --eps_safe 0.15 --pos_fraction=0.3 --RCPO --lambda 50 --logdir maze --logdir_suffix RCPO --num_eps 500 --seed $i
done

