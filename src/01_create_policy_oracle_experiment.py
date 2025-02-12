"""
Creating a Behavior Oracle experiment directory and initializing it with defaults
"""

import os
import argparse

from lib.config import Config
from lib.logger import Logger, print_
import lib.utils as utils
from CONFIG import CONFIG



def get_oracle_arguments():
    """
    Processing arguments for 01_create_behavior_experiment.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="SAVi exp-directory where the Oracle exp. will be created",
            required=True
        )
    parser.add_argument(
            "--name",
            help="Name to give to the Oracle experiment",
            required=True
        )
    args = parser.parse_args()
    args.exp_directory = utils.process_experiment_directory_argument(
                args.exp_directory,
                create=True
            )
    return args


def create_oracle_experiment():
    """
    Creating a Behavior Oracle experiment directory and initializing it with defaults
    """
    # reading command line args
    args = get_oracle_arguments()
    exp_dir, exp_name = args.exp_directory, args.name

    # making sure everything adds up
    base_exp_path = CONFIG["paths"]["experiments_path"]
    parent_path = os.path.join(base_exp_path, exp_dir)
    exp_path = os.path.join(base_exp_path, exp_dir, "oracle", exp_name)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"{parent_path = } does not exist")
    if not os.path.exists(os.path.join(parent_path, "experiment_params.json")):
        raise FileNotFoundError(f"{parent_path = } does not have experiment_params...")
    if os.path.exists(exp_path):
        raise ValueError(f"{exp_path = } already exists. Choose a different name!")

    # creating directories
    utils.create_directory(exp_path)
    _ = Logger(exp_path)  # initialize logger once exp_dir is created
    utils.create_directory(dir_path=exp_path, dir_name="plots")
    utils.create_directory(dir_path=exp_path, dir_name="tboard_logs")

    # adding experiment parameters from the parent directory, but only with specified behavior params
    try:
        cfg = Config(exp_path=parent_path)
        exp_params = cfg.load_exp_config_file()
        exp_params = cfg.add_oracle_parameters(exp_params=exp_params)
        cfg.save_exp_config_file(exp_path=exp_path, exp_params=exp_params)
    except Exception as e:
        print_("An error has occurred...\n Removing Oracle experiment directory")
        utils.delete_directory(dir_path=exp_path)
        print(e)
        exit()
    print(f"Oracle experiment {exp_name} created successfully! :)")
    return



if __name__ == "__main__":
    utils.clear_cmd()
    create_oracle_experiment()

