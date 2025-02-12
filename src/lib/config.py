"""
Methods to manage parameters and configurations
"""

import os
import json

from configs import get_config
from lib.logger import print_
from lib.utils import timestamp
import configs
from CONFIG import DEFAULTS



class Config(dict):
    """
    Main module to initialize, save, load, and process the experiment parameters.
    """
    _default_values = DEFAULTS
    _config_groups = ["dataset", "model", "training", "loss"]


    def __init__(self, exp_path):
        """
        Populating the dictionary with the default values
        """
        for key in self._default_values.keys():
            self[key] = self._default_values[key]
        self["_general"] = {}
        self["_general"]["exp_path"] = exp_path
        return


    def create_exp_config_file(self, exp_path=None, model_name=None, dataset_name=None):
        """
        Creating a JSON file with exp configs in the experiment path
        """
        exp_path = exp_path if exp_path is not None else self["_general"]["exp_path"]
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"ERROR!: exp_path {exp_path} does not exist...")

        # creating from defaults, given the model name
        else:
            for key in Config._default_values.keys():
                if key == "model":
                    self["model"] = configs.get_model_config(model_name)
                elif key == "dataset":
                    self["dataset"] = configs.get_dataset_config(dataset_name)
                elif key in ["prediction_params", "prediction_params", "predictor_loss"]:
                    _ = self.pop(key)
                    continue
                else:
                    self[key] = Config._default_values[key]

        # updating general and saving
        self["_general"]["created_time"] = timestamp()
        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(self, file)
        return


    def load_exp_config_file(self, exp_path=None, fname="experiment_params.json"):
        """
        Loading the JSON file with exp configs
        """
        if exp_path is not None:
            self["_general"]["exp_path"] = exp_path
        exp_config = os.path.join(self["_general"]["exp_path"], fname)
        if not os.path.exists(exp_config):
            raise FileNotFoundError(f"ERROR! exp. configs file {exp_config} does not exist...")

        with open(exp_config) as file:
            self = json.load(file)
        return self


    def update_config(self, exp_params):
        """
        Updating an experiments parameters file with newly added configurations from CONFIG.
        """
        for group in Config._config_groups:
            if not isinstance(Config._default_values[group], dict):
                continue
            for k in Config._default_values[group].keys():
                if(k not in exp_params[group]):
                    if(isinstance(Config._default_values[group][k], (dict))):
                        exp_params[group][k] = {}
                    else:
                        exp_params[group][k] = Config._default_values[group][k]

                if(isinstance(Config._default_values[group][k], dict)):
                    for q in Config._default_values[group][k].keys():
                        if(q not in exp_params[group][k]):
                            exp_params[group][k][q] = Config._default_values[group][k][q]
        return exp_params


    def save_exp_config_file(self, exp_path=None, exp_params=None):
        """
        Dumping experiment parameters into path
        """
        exp_path = self["_general"]["exp_path"] if exp_path is None else exp_path
        exp_params = self if exp_params is None else exp_params

        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(exp_params, file)
        return


    def add_predictor_parameters(self, exp_params, predictor_name):
        """
        Adding predictor parameters and predictor training meta-parameters
        to exp_params dictionary, and removing some unnecessaty keys
        """
        # adding predictor parameters
        predictor_params = get_config(key="predictors", name=predictor_name)
        exp_params["predictor"] = predictor_params

        # adding predictor training and predictor loss parameteres
        exp_params["prediction_params"] = DEFAULTS["prediction_params"]
        exp_params["predictor_loss"] = DEFAULTS["predictor_loss"]

        # reodering exp-params to have the desired key orderign
        sorted_keys = ["dataset", "model", "predictor", "predictor_loss", "training",
                       "prediction_params", "_general"]
        exp_params = {k: exp_params[k] for k in sorted_keys}
        return exp_params


    def add_behavior_parameters(self, exp_params):
        """ 
        Adding Action-Decoding and Behavior-Cloning params to exp_params dict
        """
        # adding behavior model and action decoder parameters
        action_decoder_params = get_config(
                key="action_decoders",
                name="MLPDecoder"
            )
        exp_params["action_decoder"] = action_decoder_params
        behavior_model_params = get_config(
                key="behavior_models",
                name="MarkovBehaviorCloner"
            )
        exp_params["behavior_model"] = behavior_model_params

        # removing predictor loss and adding loss for learning behavior            
        if "predictor_loss" in exp_params.keys():
            _ = exp_params.pop("predictor_loss")
        exp_params["loss"] = [                
                {
                    "type": "latent_action_mse",  # for behavior cloning
                    "weight": 1
                },
                {
                    "type": "action_mse",  # for action decoding
                    "weight": 0.01
                }
            ]

        # reodering exp-params to have the desired key orderign
        sorted_keys = [
            "dataset",
            "model", "predictor", "behavior_model", "action_decoder", 
            "loss", "training",
            "_general"
        ]
        
        print_(f"Creating new Exp-Params...")
        new_exp_params = {}
        for k in sorted_keys:
            if k in exp_params.keys():
                new_exp_params[k] = exp_params[k]
            else:
                print_(f"  --> Key '{k}' could not be added")
    
        return new_exp_params


    def add_oracle_parameters(self, exp_params):
        """ 
        Adding Behavior-Cloning params to exp_params dict for Oracle training
        """
        behavior_model_params = get_config(
                key="behavior_models",
                name="MarkovBehaviorCloner"
            )
        exp_params["behavior_model"] = behavior_model_params

        exp_params["loss"] = [                
                {
                    "type": "latent_action_mse",
                    "weight": 1
                }
            ]

        # reodering exp-params to have the desired key orderign
        sorted_keys = [
            "dataset",
            "model", "behavior_model",
            "loss", "training",
            "_general"
        ]
        
        print_(f"Creating new Exp-Params...")
        new_exp_params = {}
        for k in sorted_keys:
            if k in exp_params.keys():
                new_exp_params[k] = exp_params[k]
            else:
                print_(f"  --> Key '{k}' could not be added")
    
        return new_exp_params



