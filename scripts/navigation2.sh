#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..10}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name navigation2 --use_recovery --MF_recovery --gamma_safe 0.65 --eps_safe 0.2 --logdir navigation2 --logdir_suffix RRL_MF --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name navigation2 --use_recovery --gamma_safe 0.65 --eps_safe 0.2 --logdir navigation2 --logdir_suffix RRL_MB --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Unconstrained
for i in {1..10}
do
	echo "Unconstrained Run $i"
	python -m rrl_main --env-name navigation2 --cuda --logdir navigation2 --logdir_suffix unconstrained --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Lagrangian Relaxation
for i in {1..10}
do
	echo "LR Run $i"
	python -m rrl_main --cuda --env-name navigation2 --gamma_safe 0.65 --eps_safe 0.2 --DGD_constraints --nu 1000 --update_nu --logdir navigation2 --logdir_suffix LR --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# RSPO
for i in {1..10}
do
	echo "RSPO Run $i"
	python -m rrl_main --cuda --env-name navigation2 --gamma_safe 0.65 --eps_safe 0.2 --DGD_constraints --nu_schedule --nu_start 2000 --logdir navigation2 --logdir_suffix RSPO --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# SQRL
for i in {1..10}
do
	echo "SQRL Run $i"
	python -m rrl_main --cuda --env-name navigation2 --gamma_safe 0.65 --eps_safe 0.2 --DGD_constraints --use_constraint_sampling --nu 1000 --update_nu --logdir navigation2 --logdir_suffix SQRL --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Reward Penalty
for i in {1..10}
do
	echo "RP Run $i"
	python -m rrl_main --env-name navigation2 --cuda --constraint_reward_penalty 3000  --logdir navigation2 --logdir_suffix RP --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# RCPO
for i in {1..10}
do
	echo "RCPO Run $i"
	python -m rrl_main --cuda --env-name navigation2 --gamma_safe 0.65 --eps_safe 0.2 --RCPO --lambda 5000 --logdir navigation2 --logdir_suffix RCPO --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

