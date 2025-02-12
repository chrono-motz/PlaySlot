"""
Creating an experiment for training a Policy Model to learn a behavior from
expert demonstrations, as well learning an Action decoder to map latent
actions into the actual robot/simulation action space.
"""

import argparse
import os

from lib.config import Config
from lib.logger import Logger, print_
import lib.utils as utils
from CONFIG import CONFIG



def fetch_experiment_arguments():
    """
    Processing arguments for 01_create_policy_experiment.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Predictor exp. directory where the behavior exp. will be created",
            required=True
        )
    parser.add_argument(
            "--name",
            help="Name to give to the behavior experiment",
            required=True
        )
    args = parser.parse_args()
    args.exp_directory = utils.process_experiment_directory_argument(
            args.exp_directory,
            create=True
        )
    if args.exp_directory[-1] == "/":
        args.exp_directory = args.exp_directory[:-1]
    return args


def initialize_experiment():
    """
    Creating action-decoding experiment directory and initializing it with defauls
    """
    # reading command line args
    args = fetch_experiment_arguments()
    exp_dir, exp_name = args.exp_directory, args.name

    # making sure everything adds up
    base_exp_path = CONFIG["paths"]["experiments_path"]
    parent_path = os.path.join(base_exp_path, exp_dir)
    exp_path = os.path.join(base_exp_path, exp_dir, "behaviors", exp_name)
    
    if os.path.basename(os.path.dirname(exp_dir)) != "predictors":
        raise ValueError(f"{exp_dir} must be a valid predictor experiment directory...")
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"{parent_path = } does not exist")
    if not os.path.exists(os.path.join(parent_path, "experiment_params.json")):
        raise FileNotFoundError(f"{parent_path = } does not have experiment_params...")
    if len(os.listdir(os.path.join(parent_path, "models"))) <= 0:
        raise FileNotFoundError("Predictor models-dir does not contain any models!...")
    if os.path.exists(exp_path):
        raise ValueError(f"{exp_path = } already exists. Choose a different name!")

    # creating directories
    utils.create_directory(exp_path)
    _ = Logger(exp_path)  # initialize logger once exp_dir is created
    utils.create_directory(dir_path=exp_path, dir_name="plots")
    utils.create_directory(dir_path=exp_path, dir_name="tboard_logs")

    # adding exp_params from the parent directory as well as action decoder
    try:
        cfg = Config(exp_path=parent_path)
        exp_params = cfg.load_exp_config_file()
        exp_params = cfg.add_behavior_parameters(exp_params=exp_params)
        cfg.save_exp_config_file(exp_path=exp_path, exp_params=exp_params)
    except Exception as e:
        print_("An error has occurred...\n Removing experiment directory")
        utils.delete_directory(dir_path=exp_path)
        print(e)
        exit()
    print(f"Action-Decoding experiment {exp_name} created successfully! :)")
    return



if __name__ == "__main__":
    utils.clear_cmd()
    initialize_experiment()

#