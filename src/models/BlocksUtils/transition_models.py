"""
Transition models used to provide a slot initialization at the subsequent time step
"""

import torch.nn as nn

from lib.logger import print_
from models.BlocksUtils.attention import TransformerEncoderBlock
from configs import get_available_configs


TRANSITION_MODULES = [
            None, "", "None", "none",  # no trasition module is used
            "TransformerBlock"         # Single-Transformer is used, as in SAVi
    ]



def get_transition_module(model_name, **kwargs):
    f"""
    Fetching the transition module to use to provide the initial slot state
    at the subsequent time step.

    Args:
    -----
    model: string
        Type of transition module to use.
    """
    if model_name not in TRANSITION_MODULES:
        raise ValueError(f"""
                Unknown transition module: {model_name = }...
                Use one of {TRANSITION_MODULES = }.
            """)

    if model_name in [None, "", "None", "none"]:
        transitor = nn.Identity()
    elif model_name == "TransformerBlock":
        _ = kwargs.pop("num_slots", None)
        transitor = TransformerEncoderBlock(
                embed_dim=kwargs.pop("slot_dim"),
                **kwargs
            )
    else:
        raise ValueError(f"""
                UPSI, {model_name = } should not have reached here...
                Use one of {TRANSITION_MODULES = }...
            """)

    print_("Transition Module:")
    print_(f"  --> model-name: {model_name}")
    for k, v in kwargs.items():
        print_(f"  --> {k}: {v}")
    return transitor


