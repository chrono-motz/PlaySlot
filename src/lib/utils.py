"""
Utils methods for bunch of purposes, including
    - Reading/writing files
    - Creating directories
    - Timestamp
    - Handling tensorboard
"""

import os
import pickle
import shutil
import random
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.logger import log_function
from CONFIG import CONFIG



#########################
# PROCESSING ARGUMENTS #
#########################


def process_experiment_directory_argument(exp_directory, create=False):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """
    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    split_exp_dir = split_path(exp_directory)
    if os.path.basename(exp_path) == split_exp_dir[0]:
        exp_directory = "/".join(split_exp_dir[1:])

    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory) and create is False):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()
    elif(not os.path.exists(exp_directory) and create is True):
        os.makedirs(exp_directory)

    return exp_directory


def process_checkpoint_argument(exp_path, checkpoint):
    """
    Making sure checkpoint exists
    """
    if checkpoint is not None:
        checkpoint_path = os.path.join(exp_path, "models", checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist in exp {exp_path}")
    return checkpoint



def process_predictor_experiment(exp_directory, name_predictor_experiment):
    """
    If the 'exp_directory' is contained in 'name_predictor_experiment', we remove the 
    former from the latter.
    """
    if exp_directory in name_predictor_experiment:
        name_predictor_experiment = name_predictor_experiment[len(exp_directory)+1:]
    dirname = "predictors"
    if not name_predictor_experiment.startswith(f"{dirname}/"):
        name_predictor_experiment = f"{dirname}/{name_predictor_experiment}"
    pred_exp_path = os.path.join(exp_directory, name_predictor_experiment)
    if not os.path.exists(pred_exp_path):
        raise FileNotFoundError(f"{pred_exp_path = } does not exist...")
    return name_predictor_experiment



def process_predictor_checkpoint(exp_path, name_predictor_experiment, checkpoint):
    """
    Making sure checkpoint exists
    """
    if checkpoint is not None:
        ckpt_path = os.path.join(
                exp_path,
                name_predictor_experiment,
                "models",
                checkpoint
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{checkpoint = } does not exist in {ckpt_path}")
    return checkpoint



def process_behavior_experiment(exp_directory, name_behavior_experiment):
    """
    If the 'exp_directory' is contained in 'name_behavior_experiment', we remove the 
    former from the latter.
    """
    if exp_directory in name_behavior_experiment:
        name_behavior_experiment = name_behavior_experiment[len(exp_directory)+1:]
    dirname = "behaviors"
    if not name_behavior_experiment.startswith(f"{dirname}/"):
        name_behavior_experiment = f"{dirname}/{name_behavior_experiment}"
    pred_exp_path = os.path.join(exp_directory, name_behavior_experiment)
    if not os.path.exists(pred_exp_path):
        raise FileNotFoundError(f"{pred_exp_path = } does not exist...")
    return name_behavior_experiment




###############
# OTHER UTILS #
###############


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = CONFIG["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def clear_cmd():
    """Clearning command line window"""
    os.system('cls' if os.name == 'nt' else 'clear')
    return


@log_function
def create_directory(dir_path, dir_name=None):
    """
    Creating a folder in given path.
    """
    if(dir_name is not None):
        dir_path = os.path.join(dir_path, dir_name)
    if(not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    return


def delete_directory(dir_path):
    """
    Deleting a directory and all its contents
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    return


def split_path(path):
    """ Splitting a path into a list containing the names of all directories """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def timestamp():
    """
    Obtaining the current timestamp in an human-readable way
    """
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


@log_function
def log_architecture(model, exp_path, fname="model_architecture.txt"):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # getting all_params
    with open(savepath, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if(isinstance(layer, torch.nn.Module)):
            log_module(module=layer, exp_path=exp_path, fname=fname)
    return


def log_module(module, exp_path, fname="model_architecture.txt", append=True):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # writing from scratch or appending to existing file
    if (append is False):
        with open(savepath, "w") as f:
            f.write("")
    else:
        with open(savepath, "a") as f:
            f.write("\n\n")

    # writing info
    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"{module.__class__.__name__}\n")
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))
    return



def remove_nans(matrix):
    """ HACK: removing rows with NaNs. Not sure where this NaNs come from though """
    nan_mask = torch.isnan(matrix).any(dim=1)
    matrix = matrix[~nan_mask]
    return matrix, nan_mask



class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    Args:
    -----
    logdir: string
        path where the tensorboard logs will be stored
    """

    def __init__(self, logdir):
        """ Initializing tensorboard writer """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        """ Adding a scalar for plot """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_images(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_images(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure, step):
        """ Adding a whole new figure to the tensorboard """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return

    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return

#
