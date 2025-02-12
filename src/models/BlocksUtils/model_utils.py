"""
Model utils
"""

import numpy as np
import torch
import torch.nn as nn

from lib.logger import print_


def build_grid(resolution, vmin=-1., vmax=1., device=None):
    """
    Building four grids with gradients [-1, 1] in directios (x, -x, y, -y)
    This can be used as a positional encoding in SAVi's encoder and decoder

    Args:
    -----
    resolution: list/tuple of integers
        number of elements in each of the gradients

    Returns:
    -------
    torch_grid: torch Tensor
        Grid gradients in 4 directions. Shape is [R, R, 4]
    """
    ranges = [np.linspace(vmin, vmax, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    torch_grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)
    return torch_grid



def count_model_params(model, verbose=False):
    """
    Counting number of learnable parameters
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print_(f"  --> Number of learnable parameters: {num_params}")
    return num_params



def freeze_params(model):
    """
    Freezing model params to avoid updates in backward pass
    """
    print_(f"Freezing {model.__class__.__name__}")
    for param in model.parameters():
        param.requires_grad = False
    return



def unfreeze_params(model):
    """
    Unfreezing model params to allow for updates during backward pass
    """
    print_(f"Unfreezing {model.__class__.__name__}")
    for param in model.parameters():
        param.requires_grad = True
    return



@torch.no_grad()
def init_xavier_(model: nn.Module):
    """
    Initializes (in-place) a model's weights with xavier uniform, and its biases to zero.
    All parameters with name containing "bias" are initialized to zero.
    All other parameters are initialized with xavier uniform with default parameters,
    unless they have dimensionality <= 1.
    """
    print_("Xavier Initialization of model parameters:")
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_() + 1e-4
        elif len(tensor.shape) <= 1:
            pass  # silent
        else:
            try:
                nn.init.xavier_uniform_(tensor)
            except:  # some tensors (e.g. CausalMask in GPT need no init)
                print_(f" --> {name} of shape {tensor.shape} could not be initialized...")
                continue



@torch.no_grad()
def init_kaiming_(model: nn.Module):
    """
    Initializes (in-place) a model's weights with kaiming uniform, and its biases to zero.
    All parameters with name containing "bias" are initialized to zero.
    All other parameters are initialized with xavier uniform with default parameters,
    unless they have dimensionality <= 1.
    """
    print_("Kaiming Initialization of model parameters:")
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_() + 1e-4
        elif hasattr(tensor, 'weight') and tensor.weight.dim() > 1:
            try:
                nn.init.kaiming_uniform(tensor)
            except:  # some tensors (e.g. CausalMask in GPT need no init)
                print_(f" --> {name} of shape {tensor.shape} could not be initialized...")
                continue



def build_slot_causal_mask(seq_len, num_slots, device):
    """
    Obtaining a binary maskign pattern for slot-based models
    The mask is a block-diagonal mask, i.e., diagona but with a 'staircase'
    """
    num_tokens = seq_len * num_slots
    mask = torch.zeros((num_tokens, num_tokens), device=device)
    for i in range(seq_len):
        mask[num_slots*i:, num_slots*i:num_slots*(i + 1)] = 1.
    return mask


#
