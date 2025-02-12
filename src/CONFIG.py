"""
Global configurations
"""

import os

###############
### CONFIGS ###
###############

# hack to fix paths when run from notebooks
project_path = os.getcwd()
while os.path.basename(project_path) != "ConditionedOCVP":
    project_path = os.path.dirname(project_path)


CONFIG = {
    "random_seed": 13,
    "num_workers": 8,
    "paths": {
        "data_path": os.path.join(project_path, "..", "..", "datasets"),
        "experiments_path": os.path.join(project_path, "experiments"),
        "configs_path": os.path.join(project_path, "src", "configs"),
        "base_path": os.path.join(project_path, "src", "base"),
        "resources_path": os.path.join(project_path, "resources")
    }
}



#######################
### DEFAULT CONFIGS ###
#######################

DEFAULTS = {
    "dataset": {
        "dataset_name": "OBJ3D",
        "shuffle_train": True,
        "shuffle_eval": False,
        "use_segmentation": True,
        "target": "rgb",
        "random_start": True
    },
    "model": {
        "model_name": "",
        "model_params": {}
    },
    "loss": [
        {
            "type": "mse",
            "weight": 1
        }
    ],
    "predictor_loss": [
        {
            "type": "pred_img_mse",
            "weight": 1
        },
        {
            "type": "pred_slot_mse",
            "weight": 1
        },
        {
            "type": "VQLoss",
            "weight": 1,
            "beta": 0.25
        }    
    ],
    "training": {  # training related parameters
        "num_epochs": 1000,        # number of epochs to train for
        "save_frequency": 10,   # saving a checkpoint after these iterations ()
        "log_frequency": 100,     # logging stats after this amount of updates
        "image_log_frequency": 100,     # logging stats after this amount of updates
        "batch_size": 64,
        "train_iters_per_epoch": 1000,
        "valid_iters_per_epoch": 100,
        "lr": 1e-4,
        "scheduler": "cosine_annealing",             # learning rate scheduler parameters
        "scheduler_steps": 300000,
        "lr_warmup": True,       # learning rate warmup parameters (2 epochs or 200 iters default)
        "warmup_steps": 4000,
        "gradient_clipping": True,
        "clipping_max_value": 0.05  # according to SAVI paper
    },
    "prediction_params": {
        "num_context": 6,
        "num_preds": 8,
        "teacher_force": False,
        "sample_length": 14,
        "input_buffer_size": 30,
    }
}


##############
### OTHERS ###
##############

# Colors and visualizations
COLORS = ["white", "blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "darkviolet", "springgreen",
          "aqua", "royalblue", "navy", "forestgreen", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]

COLOR_MAP = {
        "context": "green",
        "targets": "blue",
        "preds": "red",
    }


#
