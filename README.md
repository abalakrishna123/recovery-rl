# Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones
------------

<p align=center>
  <img src="img/recovery-rl-simple_website.png" width=800>
  <img src="img/domains_website.png" width=800>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2010.15920.pdf">View on ArXiv</a>
  |
  <a href="https://sites.google.com/berkeley.edu/recovery-rl/">View website</a>
</p>

# Description
------------
Implementation of  <a href="https://arxiv.org/pdf/2010.15920.pdf">Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones</a>. The SAC code is built 
on top of the Pytorch implementation of Soft Actor Critic from <a href="https://github.com/pranz24/pytorch-soft-actor-critic">pytorch-soft-actor-critic</a>. For the 
recovery policy, we build on the implementations of the PETS algorithm from <a href="https://github.com/quanvuong/handful-of-trials-pytorch">handful-of-trials-pytorch.</a> 
and the latent visual dynamics model from <a href="https://github.com/suraj-nair-1/goal_aware_prediction">goal_aware_prediction</a>.

The repository is organized as follows. The env folder contains implementations of all environments used in the paper while the config folder contains the environment
specific parameters used for the learrned recovery policies. The recovery_rl folder contains the core implementation of the Recovery RL algorithm. The SAC implementation
can be found in `SAC.py`, while the safety critic and model-free recovery policy implementation can be found in `qrisk.py`. The model-based recovery policy is implemented in 
`MPC.py` and `optimizers.py` (for low-dimensional experiments) and `VisualMPC.py` (for image-based experiments). We also include implementations of the core neural network
modules used for all approaches in `model.py`, the replay buffer used for training in `replay_memory.py` and general utilities in `utils.py`. Finally, we include an experiment wrapper in `experiment.py` to create and run experiments and log results.

The main script for running experiments is `rrl_main.py` in the root direrctory, which parses command-line arguments from the user using the options in `arg_utils.py`, instantiates an experiment with the experiment wrapper in `recovery_rl/experiment.py`, and runs the experiment. 

# Installation and Setup
------------
For installation, run `. install.sh`. This will install all python and system wide dependencies for Recovery RL and also download
(1) the offline data needed for recovery policy training and (2) a pre-trained visual dynamics model for visual model based recovery for the Image Maze environment.

# Running Experiments
------------
We include all code to replicate experiments for the Recovery RL paper (Recovery RL algorithm and all 6 baseline algorithms) in the scripts folder. Use the following scripts to replicate results for each of the experimental domains in the paper.

### Navigation 1
`. scripts/navigation1.sh`

###  Navigation 2
`. scripts/navigation2.sh`

###  Maze
`. scripts/maze.sh`

###  Image Maze
`. scripts/image_maze.sh`

###  Object Extraction
`. scripts/obj_extraction.sh`

###  Object Extraction (Dynamic Obstacle)
`. scripts/obj_dynamic_extraction.sh`

Ablations and Sensitivity Experiments:

###  Ablations
`. scripts/ablations.sh`

###  Sensitivity Experiments
`. scripts/ablations.sh`

# Plotting Results
------------
Update PLOT_TYPE on the top of the file. Use 'ratio' to replicate the main plots in the paper, 'success' to only visualize cumulative tasks succcesses, and 'violation' to only visualize cumulative task successes. The 'PR' option can be used to replicate the sensitivity experiments plot in the paper while the 'reward' option can be used to replicate the learning curves in the supplementary material. To plot results, update the experiment name and log directory at the bottom of the file and run `python plotting/plot_runs.py`
