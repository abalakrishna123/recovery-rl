import torch
import numpy as np

'''
Wrapper for querying a Recovery RL Agent
'''


class RecoveryRLAgent:
    def __init__(self, task_policy, recovery_policy, env,
                 args, total_numsteps):
        
        self.task_policy = task_policy
        self.recovery_policy = recovery_policy
        self.env = env 
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.args = args 
        self.total_numsteps = total_numsteps
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)

    def sample(self, state, train=True):
        if self.args.start_steps > self.total_numsteps and train:
            actions = self.torchify(
                        np.random.uniform(self.ac_lb, self.ac_ub, (state.shape[0], self.ac_lb.shape[0])))
        elif train:
            actions = self.task_policy.select_action(
                state)  # Sample action from policy
        else:
            actions = self.task_policy.select_action(
                state, eval=True)  # Sample action from policy

        q_risk_values = self.task_policy.safety_critic.get_value(
            state, actions)

        recovery = q_risk_values <= self.args.eps_safe
        if self.args.MF_recovery or self.args.Q_sampling_recovery:
            real_actions = torch.where(
                recovery,
                self.task_policy.safety_critic.select_action(state),
                actions
            )
        else:
            real_actions = torch.where(
                recovery,
                self.torchify(self.recovery_policy.act(state, 0)),
                actions
            )

        actions = self.torchify(actions.detach().cpu().numpy())
        real_actions = self.torchify(real_actions.detach().cpu().numpy())
        return actions, real_actions, recovery

