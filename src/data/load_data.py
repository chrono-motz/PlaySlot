"""
Methods for loading specific datasets, fitting data loaders and other
data utils functionalities
"""

import os
from torch.utils.data import DataLoader        
from CONFIG import CONFIG
from configs import get_available_configs
from lib.logger import print_



def load_data(exp_params, split="train"):
    """
    Loading a dataset given the parameters

    Args:
    -----
    dataset_name: string
        name of the dataset to load
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    """
    db_params = exp_params["dataset"]
    dataset_name = db_params["dataset_name"]
    DATASETS = get_available_configs("datasets")    
    if dataset_name not in DATASETS:
        raise NotImplementedError(
                f"""ERROR! Dataset'{dataset_name}' is not available.
                Please use one of the following: {DATASETS}..."""
            )

    # Block-based Object-Centric datasets (and SynPick)
    if dataset_name == "GridShapes":
        from data.GridShapes import GridShapes
        dataset = GridShapes(split=split, **db_params)
    # BlockPush Datasets
    elif dataset_name == "BlockPush":
        from data.BlockPush import BlockPush
        dataset = BlockPush(split=split, **db_params)
    elif dataset_name == "BlockPush_ExpertDemos":
        from data.BlockPush import BlockPushExpertDemos
        dataset = BlockPushExpertDemos(split=split, **db_params)
    # Sketchy Dataset
    elif dataset_name == "Sketchy":
        from data.Sketchy import Sketchy
        dataset = Sketchy(split=split, **db_params)
    # MetaWorld ButtonPress Datasets
    elif dataset_name == "ButtonPress":
        from data.ButtonPress import ButtonPress
        dataset = ButtonPress(split=split, **db_params)
    elif dataset_name == "ButtonPress_ExpertDemos":
        from data.ButtonPress import ButtonPressExpertDemos
        dataset = ButtonPressExpertDemos(split=split, **db_params)
    else:
        raise NotImplementedError(
                f"""ERROR! Dataset'{dataset_name}' is not available.
                Please use one of the following: {DATASETS}..."""
            )
    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"]
        )
    return data_loader


def unwrap_batch_data(exp_params, batch_data):
    """
    Unwrapping the batch data depending on the dataset that we are training on
    """
    initializer_kwargs = {}
    others = {}
    if exp_params["dataset"]["dataset_name"] in ["GridShapes"]:
        videos, targets, meta = batch_data
        others = {**others, **meta}
    elif exp_params["dataset"]["dataset_name"] in [
                "ButtonPress", "ButtonPress_ExpertDemos",
                "BlockPush", "BlockPush_ExpertDemos",
                "Sketchy"
            ]: 
        videos, targets, all_reps = batch_data
        others["actions"] = all_reps.get("actions")
    else:
        dataset_name = exp_params["dataset"]["dataset_name"]
        raise NotImplementedError(f"Dataset {dataset_name} is not supported...")
    return videos, targets, initializer_kwargs, others


def set_expert_policy_dataset(db_params):
    """ 
    Exchanging the current dataset with a variant using an ExpertPolicy
    """
    dataset_name = db_params["dataset_name"]
    if "ButtonPress" in dataset_name:
        updated_db_params = set_expert_policy_buttonpress(db_params)
    elif "BlockPush" in dataset_name:
        updated_db_params = set_expert_policy_blockpush(db_params)
    else:
        raise NameError("We cannot set expert policy to {dataset_name = }")
    return updated_db_params


def set_expert_policy_blockpush(db_params):
    """
    Exchanging BlockPush with the variant generated using an expert policy
    """
    dataset_name = db_params["dataset_name"]
    if "BlockPush" not in dataset_name:
        raise NameError(f"{dataset_name = } must be a 'BlockPush' variant...")
    print_(f"Using Expert Policy dataset variant:")
    
    print_(f"  -->Exchanging db from {dataset_name} to 'BlockPush_ExpertDemos'...")
    db_params["dataset_name"] = "BlockPush_ExpertDemos"    
    db_params["datapath"] = os.path.join(
            os.path.dirname(db_params["datapath"]),
            "Expert_BlockPush"
        )
    if "num_expert_demos" not in db_params:
        db_params["num_expert_demos"] = -1
    return db_params


def set_expert_policy_buttonpress(db_params):
    """
    Exchanging ButtonPress dataset with the variant generated using an expert policy
    """
    dataset_name = db_params["dataset_name"]
    if "ButtonPress" not in dataset_name:
        raise NameError(f"{dataset_name = } must be a 'ButtonPress' variant...")
    print_(f"Using Expert Policy dataset variant:")
    
    print_(f"  -->Exchanging db from {dataset_name} to 'ButtonPress_ExpertDemos'...")
    db_params["dataset_name"] = "ButtonPress_ExpertDemos"
    db_params["datapath"] = os.path.join(
            os.path.dirname(db_params["datapath"]),
            "Expert_ButtonPress"
        )    
    if "num_expert_demos" not in db_params:
        db_params["num_expert_demos"] = -1
    return db_params


#
