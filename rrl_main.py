from arg_utils import get_args
from recovery_rl.experiment import Experiment

if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()
    # Create experiment and run it
    experiment = Experiment(exp_cfg)
    experiment.run()
