#!/bin/bash

# ----- # Demos Ablations -----

# 100 constraint demos
for i in {1..10}
do
	echo "RRL 100 Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 100 --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_100 --num_eps 4000 --seed $i
done

# # 500 constraint demos
for i in {1..10}
do
	echo "RRL 500 Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 500 --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_500 --num_eps 4000 --seed $i
done

# # 1000 constraint demos
for i in {1..10}
do
	echo "RRL 1000 Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 1000 --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_1000 --num_eps 4000 --seed $i
done

# # 5000 constraint demos
for i in {1..10}
do
	echo "RRL 5000 Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --num_constraint_transitions 5000 --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_5000 --num_eps 4000 --seed $i
done

# # ----- # Method Ablations -----

# # Disable Action Relabeling
for i in {1..10}
do
	echo "RRL (-relabel) Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --disable_action_relabeling --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_relabel --num_eps 4000 --seed $i
done

# # Disable Online Updates
for i in {1..10}
do
	echo "RRL (-online) Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --disable_online_updates --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_online --num_eps 4000 --seed $i
done

# # Disable Offline Updates
for i in {1..10}
do
	echo "RRL (-offline) Run $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --disable_offline_updates --pos_fraction 0.3 --logdir ablations --logdir_suffix RRL_offline --num_eps 4000 --seed $i
done

