#!/bin/bash

# Recovery RL (model-based recovery) 0.75, 0.15
for i in {1..10}
do
	echo "RRL MB Run 0.75 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.15 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.75_0.15 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.75, 0.25
for i in {1..10}
do
	echo "RRL MB Run 0.75 0.25 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.75_0.25 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.75, 0.35
for i in {1..10}
do
	echo "RRL MB Run 0.75 0.35 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.35 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.75_0.35 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.75, 0.45
for i in {1..10}
do
	echo "RRL MB Run 0.75 0.45 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.45 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.75_0.45 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.75, 0.55
for i in {1..10}
do
	echo "RRL MB Run 0.75 0.55 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.75 --eps_safe 0.55 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.75_0.55 --num_eps 4000 --seed $i
done





# # Recovery RL (model-based recovery) 0.85, 0.15
for i in {1..10}
do
	echo "RRL MB Run 0.85 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.15 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.85_0.15 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.85, 0.25
for i in {1..10}
do
	echo "RRL MB Run 0.85 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.25 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.85_0.25 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.85, 0.35
for i in {1..10}
do
	echo "RRL MB Run 0.85 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.35 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.85_0.35 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.85, 0.45
for i in {1..10}
do
	echo "RRL MB Run 0.85 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.45 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.85_0.45 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.85, 0.55
for i in {1..10}
do
	echo "RRL MB Run 0.85 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.85 --eps_safe 0.55 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.85_0.55 --num_eps 4000 --seed $i
done





# Recovery RL (model-based recovery) 0.95, 0.15
for i in {1..10}
do
	echo "RRL MB Run 0.95 0.15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.95 --eps_safe 0.15 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.95_0.15 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.95, 0.25
for i in {1..10}
do
	echo "RRL MB Run 0.95 0.25 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.95 --eps_safe 0.25 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.95_0.25 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.95, 0.35
for i in {1..10}
do
	echo "RRL MB Run 0.95 0.35 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.95 --eps_safe 0.35 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.95_0.35 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.95, 0.45
for i in {1..10}
do
	echo "RRL MB Run 0.95 0.45 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.95 --eps_safe 0.45 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.95_0.45 --num_eps 4000 --seed $i
done

# # Recovery RL (model-based recovery) 0.95, 0.55
for i in {1..10}
do
	echo "RRL MB Run 0.95 0.55 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --use_recovery --recovery_policy_update_freq 20 --gamma_safe 0.95 --eps_safe 0.55 --use_qvalue --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RRL_MB_0.95_0.55 --num_eps 4000 --seed $i
done




# Lagrangian Relaxation 5
for i in {1..10}
do
	echo "LR Run 5 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 5 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix LR_5 --num_eps 4000 --seed $i
done

# # Lagrangian Relaxation 10
for i in {1..10}
do
	echo "LR Run 10 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 10 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix LR_10 --num_eps 4000 --seed $i
done

# # Lagrangian Relaxation 15
for i in {1..10}
do
	echo "LR Run 15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 15 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix LR_15 --num_eps 4000 --seed $i
done

# # Lagrangian Relaxation 25
for i in {1..10}
do
	echo "LR Run 25 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 25 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix LR_25 --num_eps 4000 --seed $i
done

# # Lagrangian Relaxation 50
for i in {1..10}
do
	echo "LR Run 50 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --DGD_constraints --update_nu --nu 50 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix LR_50 --num_eps 4000 --seed $i
done




# # Reward Penalty 5
for i in {1..10}
do
	echo "RP Run 5 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 5 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RP_5 --num_eps 4000 --seed $i
done

# # Reward Penalty 10
for i in {1..10}
do
	echo "RP Run 10 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 10 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RP_10 --num_eps 4000 --seed $i
done

# # Reward Penalty 15
for i in {1..10}
do
	echo "RP Run 15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 15 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RP_15 --num_eps 4000 --seed $i
done

# # Reward Penalty 25
for i in {1..10}
do
	echo "RP Run 25 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 25 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RP_25 --num_eps 4000 --seed $i
done

# # Reward Penalty 50
for i in {1..10}
do
	echo "RP Run 50 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --constraint_reward_penalty 50 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RP_50 --num_eps 4000 --seed $i
done




# # RCPO 5
for i in {1..10}
do
	echo "RCPO Run 5 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --lambda 5 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RCPO_5 --num_eps 4000 --seed $i
done

# # RCPO 10
for i in {1..10}
do
	echo "RCPO Run 10 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --lambda 10 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RCPO_10 --num_eps 4000 --seed $i
done

# # RCPO 15
for i in {1..10}
do
	echo "RCPO Run 15 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --lambda 15 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RCPO_15 --num_eps 4000 --seed $i
done

# # RCPO 25
for i in {1..10}
do
	echo "RCPO Run 25 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --lambda 25 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RCPO_25 --num_eps 4000 --seed $i
done

# # RCPO 50
for i in {1..10}
do
	echo "RCPO Run 50 $i"
	python -m rrl_main --cuda --env-name obj_extraction --task_demos --tau 0.0002 --replay_size 100000 --num_task_transitions 1000 --RCPO --gamma_safe 0.75 --eps_safe 0.25 --use_qvalue --lambda 50 --pos_fraction 0.3 --logdir sensitivity --logdir_suffix RCPO_50 --num_eps 4000 --seed $i
done
