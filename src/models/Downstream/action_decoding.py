""" 
Modules used to decoder latent actions into the actual actions
from the robot/simulation
"""

import torch.nn as nn


class MLPActionDecoder(nn.Module):
    """ 
    Simple MLP-based action decoder
    
    Args:
    -----
    in_dim: int
        Dimensionality of the Latent Actions output by the latent action model
    out_dim: int
        Dimensionality of the ground-truth actions
    hidden_dim: list/tuple
        List with the hidden dimensionality of the MLP. It defines the number of layers
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=[128, 128]):
        """ Module initialzier """
        assert isinstance(hidden_dim, (list, tuple))
        assert len(hidden_dim) > 0
        assert out_dim > 0
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        # instanciating MLP decoder
        mlp = []
        for i in range(len(hidden_dim)):
            dim1 = in_dim if i == 0 else hidden_dim[i-1]
            dim2 = hidden_dim[i]
            mlp.append(
                    nn.Sequential(nn.Linear(dim1, dim2), nn.ReLU())
                )
        mlp.append(nn.Linear(hidden_dim[-1], out_dim))
        self.mlp = nn.Sequential(*mlp)
        return

    def forward(self, latent_action):
        """ Decoding latent actions """
        dec_action = self.mlp(latent_action)
        return dec_action
    
    
#