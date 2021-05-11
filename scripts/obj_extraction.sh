#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..10}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --gamma_safe 0.85 --eps_safe 0.35 --MF_recovery --pos_fraction 0.3 --num_unsafe_transitions 20000  --logdir obj_extraction --logdir_suffix RRL_MF --num_eps 4000 --seed $i
done

# Recovery RL (model-based recovery)
for i in {1..10}
do
	echo "RRL MB Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.25 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix RRL_MB --num_eps 4000 --seed $i
done

# Unconstrained
for i in {1..10}
do
	echo "Unconstrained Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix unconstrained --num_eps 4000 --seed $i
done

# Lagrangian Relaxation
for i in {1..10}
do
	echo "LR Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --DGD_constraints --update_nu --nu 50 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix LR --num_eps 4000 --seed $i
done

# RSPO
for i in {1..10}
do
	echo "RSPO Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --DGD_constraints --nu_schedule --nu_start 100 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix RSPO --num_eps 4000 --seed $i
done

# SQRL
for i in {1..10}
do
	echo "SQRL Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --DGD_constraints --update_nu --use_constraint_sampling --nu 50 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix SQRL --num_eps 4000 --seed $i
done

# Reward Penalty
for i in {1..10}
do
	echo "RP Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 50 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix RP --num_eps 4000 --seed $i
done

# RCPO
for i in {1..10}
do
	echo "RCPO Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --lambda 50 --pos_fraction 0.3 --num_unsafe_transitions 20000 --logdir obj_extraction --logdir_suffix RCPO --num_eps 4000 --seed $i
done
