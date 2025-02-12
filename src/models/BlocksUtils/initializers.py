"""
Modules for the initalization of the slots on Slot-Attention and SAVI
"""

import torch
import torch.nn as nn
from math import sqrt

from lib.logger import print_

INITIALIZERS = [
        "Learned",         # learning slots
        "LearnedRandom",   # learning Gaussian and sampling slots
        "CoM",             # encoding Centrer of Mass into slots
        "BBox"             # encoding bounding boxes into slots 
    ]


def get_initializer(mode, slot_dim, num_slots):
    f"""
    Fetching the initializer module for the slots

    Args:
    -----
    model: string
        Type of initializer to use. Valid modes are {INITIALIZERS}
    slot_dim: int
        Dimensionality of the slots
    num_slots: int
        Number of slots to initialize
    """
    if mode not in INITIALIZERS:
        raise ValueError(f"Unknown initializer {mode = }. Available {INITIALIZERS = }")

    if mode == "Learned":
        intializer = Learned(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "LearnedRandom":
        intializer = LearnedRandom(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "CoM":
        intializer = CoordInit(slot_dim=slot_dim, num_slots=num_slots, mode="CoM")
    elif mode == "BBox":
        intializer = CoordInit(slot_dim=slot_dim, num_slots=num_slots, mode="BBox")
    else:
        raise ValueError(f"UPSI, {mode = } should not have reached here...")

    print_("Initializer:")
    print_(f"  --> mode={mode}")
    print_(f"  --> slot_dim={slot_dim}")
    print_(f"  --> num_slots={num_slots}")
    return intializer



class Learned(nn.Module):
    """
    Learned Initialization.
    Slots are learned via backpropagation..
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the slots vectors.
    num_slots: int
        Number of slots to learn
    """

    def __init__(self, slot_dim, num_slots):
        """ Module intializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots, -limit, limit)
        self.slots.requires_grad_()
        return

    def forward(self, batch_size, **kwargs):
        """ Sampling random Gaussian slots """
        slots = self.slots.repeat(batch_size, 1, 1)
        return slots



class LearnedRandom(nn.Module):
    """
    Learned-Random intialization.
    Slots are randomly sampled from a learned Gaussian distribution
    The mean and diagonal covariance of such distribution are learned via backpropagation
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the slots vectors.
    num_slots: int
        Number of slots to learn
    """

    def __init__(self, slot_dim, num_slots):
        """ Module intializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        # learned statistics of the Gaussian distribution
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_sigma, -limit, limit)
        self.slots_mu.requires_grad_()
        self.slots_sigma.requires_grad_()
        return

    def forward(self, batch_size, **kwargs):
        """ Sampling random slots from the learned gaussian distribution """
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=self.slots_mu.device)
        return slots



class CoordInit(nn.Module):
    """
    Slots are initalized by encoding, for each object, the coordinates of one of the following:
        - the CoM of the instance segmentation of each object, represented as [y, x]
        - the BBox containing each object, represented as [y_min, x_min, y_max, x_max]
        
    Args:
    -----
    slot_dim: int
        Dimensionality of the slots vectors.
    num_slots: int
       Number of slots to learn
    mode: str  
       Annotations to encoder for the initialization. Can be either 'CoM' or 'BBox'.
    """

    MODE_REP = {
            "CoM": "com_coords",
            "BBox": "bbox_coords"
        }
    IN_FEATS = {
            "CoM": 2,
            "BBox": 4
        }

    def __init__(self, slot_dim, num_slots, mode):
        """ Module intializer """
        modes = list(CoordInit.MODE_REP.keys())
        assert mode in modes, f"Unknown {mode = }. Use one of {modes}"
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.mode = mode
        self.in_feats = CoordInit.IN_FEATS[self.mode]  # 2 for CoM and 4 for BBox

        # simple MLP for encoding CoM or BBox coordinates into slots
        self.coord_encoder = nn.Sequential(
                nn.Linear(self.in_feats, 256),
                nn.ReLU(),
                nn.Linear(256, slot_dim),
            )
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))  # to get the device
        return

    def forward(self, **kwargs):
        """ Encoding BBox or CoM coordinates into slots using an MLP """
        device = self.dummy_parameter.device
        rep_name = CoordInit.MODE_REP[self.mode]
        in_feats = CoordInit.IN_FEATS[self.mode]

        # sanity checks 
        coords = kwargs.get(rep_name, None)
        if coords is None or coords.sum() == 0:
            raise ValueError(f"{self.mode} Initializer requires '{rep_name}' as input...")
        if len(coords.shape) == 4:  # getting only coords corresponding to time-step t=0
            coords = coords[:, 0]
        coords = coords.to(device)

        # obtaining [-1]-vectors for padding the slots that currently do not have an object
        B, num_coords = coords.shape[0], coords.shape[1]
        if num_coords > self.num_slots:
            raise ValueError(f"There shouldnt be more {num_coords = } than {self.num_slots = }! ")
        if num_coords < self.num_slots:
            remaining_masks = self.num_slots - num_coords
            pad_zeros = -1 * torch.ones((B, remaining_masks, in_feats), device=device)
            coords = torch.cat([coords, pad_zeros], dim=2)

        slots = self.coord_encoder(coords)
        return slots

