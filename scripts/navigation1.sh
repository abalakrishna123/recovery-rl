#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..10}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name navigation1 --use_recovery --MF_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir navigation1 --logdir_suffix RRL_MF --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name navigation1 --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir navigation1 --logdir_suffix RRL_MB --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Unconstrained
for i in {1..10}
do
	echo "Unconstrained Run $i"
	python -m rrl_main --env-name navigation1 --cuda --logdir navigation1 --logdir_suffix unconstrained --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Lagrangian Relaxation
for i in {1..10}
do
	echo "LR Run $i"
	python -m rrl_main --cuda --env-name navigation1 --gamma_safe 0.8 --eps_safe 0.3 --DGD_constraints --nu 5000 --update_nu --logdir navigation1 --logdir_suffix LR --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# RSPO
for i in {1..10}
do
	echo "RSPO Run $i"
	python -m rrl_main --cuda --env-name navigation1 --gamma_safe 0.8 --eps_safe 0.3 --DGD_constraints --nu_schedule --nu_start 10000 --logdir navigation1 --logdir_suffix RSPO --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# SQRL
for i in {1..10}
do
	echo "SQRL Run $i"
	python -m rrl_main --cuda --env-name navigation1 --gamma_safe 0.8 --eps_safe 0.3 --DGD_constraints --use_constraint_sampling --nu 5000 --update_nu --logdir navigation1 --logdir_suffix SQRL --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Reward Penalty
for i in {1..10}
do
	echo "RP Run $i"
	python -m rrl_main --env-name navigation1 --cuda --constraint_reward_penalty 1000  --logdir navigation1 --logdir_suffix RP --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# RCPO
for i in {1..10}
do
	echo "RCPO Run $i"
	python -m rrl_main --cuda --env-name navigation1 --gamma_safe 0.8 --eps_safe 0.3 --RCPO --lambda 1000 --logdir navigation1 --logdir_suffix RCPO --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

