"""
Basic building blocks for neural nets
"""

import math
import torch
import torch.nn as nn

from models.BlocksUtils.model_utils import build_grid

__all__ = [
        "ConvBlock",
        "SoftPositionEmbed",
        "LearnedPositionalEncoding",
        "SlotPositionalEncoding",
        "SinusoidalPositionalEncoding",
        "MLP"
    ]



class ConvBlock(nn.Module):
    """
    Simple convolutional block for conv. encoders

    Args:
    -----
    in_channels: int
        Number of channels in the input feature maps.
    out_channels: int
        Number of convolutional kernels in the conv layer
    kernel_size: int
        Size of the kernel for the conv layer
    stride: int
        Amount of strid applied in the convolution
    padding: int/None
        Whether to pad the input feature maps, and how much padding to use.
    batch_norm: bool
        If True, Batch Norm is applied after the convolutional layer
    max_pool: int/tuple/None
        If not None, output feature maps are downsampled by this amount via max pooling
    activation: bool
        If True, output feature maps are activated via a ReLU nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, max_pool=None, upsample=None, activation=True):
        """ Module initializer """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if max_pool:
            assert isinstance(max_pool, (int, tuple, list))
            layers.append(nn.MaxPool2d(kernel_size=max_pool, stride=max_pool))
        if upsample is not None:
            assert isinstance(upsample, (int))
            assert max_pool is None, f"'max_pool' and 'upsample' cannot both be provided"
            layers.append(nn.Upsample(scale_factor=upsample, mode="bilinear"))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y



class SoftPositionEmbed(nn.Module):
    """
    Soft positional embedding with learnable linear projection:
        1. The positional encoding corresponds to a 4-channel grid with coords
           [-1, ..., 1] and [1, ..., -1] in the x- and y-directions
        2. The 4 channels are projected into a hidden_dimension via a 1D-convolution

    Args:
    -----
    hidden_size: int
        Number of output channels
    resolution: list/tuple of integers
        Number of elements in the positional embedding. Corresponds to a spatial size
    vmin, vmax: int
        Minimum and maximum values in the grids. By default vmin=-1 and vmax=1
    """

    def __init__(self, hidden_size, resolution, vmin=-1., vmax=1.):
        """ Soft positional encoding """
        super().__init__()
        self.resolution = resolution
        self.projection = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.grid = build_grid(resolution, vmin=vmin, vmax=vmax).permute(0, 3, 1, 2)
        return

    def forward(self, inputs, channels_last=True):
        """ Projecting grid and adding to inputs """
        b_size = inputs.shape[0]
        if self.grid.device != inputs.device:
            self.grid = self.grid.to(inputs.device)
        grid = self.grid.repeat(b_size, 1, 1, 1)
        emb_proj = self.projection(grid)
        if channels_last:
            emb_proj = emb_proj.permute(0, 2, 3, 1)
        return inputs + emb_proj



class LearnedPositionalEncoding(nn.Module):
    """
    1-Dimensional learned positional encoding
    
    Args:
    -----
    max_len: int
        Maximum number of tokens in the sequence.
    token_dim: int
        Dimension of the token embeddings.
    dropout: float
        Dropout rate.
    """

    def __init__(self, max_len, token_dim, dropout=0.1):
        """ Initializer """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, token_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pe)
        return

    def forward(self, x, offset=0):
        """ Adding positional encoding """
        T = x.shape[1]
        y = self.dropout(x + self.pe[:, offset:offset + T])
        return y



class SlotPositionalEncoding(nn.Module):
    """
    Positional encoding to be added to the input tokens of the transformer predictor.

    Our positional encoding only informs about the time-step, i.e., all slots extracted
    from the same input frame share the same positional embedding.
    This allows our predictor to maintain the permutation equivariance of the slots.

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Percentage of dropout to apply after adding the poisitional encoding. Default is 0.1.
        WARNING! Using dropout > 0 has caused poor performance in the past!
    max_len: int
        Length of the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=50):
        """ Initializing the positional encoding """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # initializing sinusoidal positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, 1, d_model)
        self.register_buffer('pe', pe)  # (1, seq_len, 1, token_dim)
        return

    def forward(self, x, batch_size, num_slots, offset=0):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding.
            Shape is (B, Seq_len, Num_Slots, Token_Dim)
        batch_size: int
            Given batch size to repeat the positional encoding for
        num_slots: int
            Number of slots to repeat the positional encoding for
        offset: int
            Temporal positions to offset in the positional encoding
        """
        if len(x.shape) != 4:
            raise ValueError(f"{x.shape = } must have 4 dimensions: (B, SeqLen, Objs, Dim)...")
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, offset:cur_seq_len+offset]
        x = x + cur_pe
        y = self.dropout(x)
        return y
    
    def get_pe(self, idx):
        """ Fetching a specific positional encoding"""
        return self.pe[:, idx, :, :]



class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Dropout to apply after adding the poisitional encoding. Default is 0.1
    max_len:t: int
        Maximum number of input tokens
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """ Initializing the positional encoding """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        return

    def forward(self, x):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding.
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        
        if len(x.shape) == 3:
            B, num_tokens, _ = x.shape
            cur_pe = self.pe.repeat(B, 1, 1)[:, :num_tokens]
        elif len(x.shape) == 4:
            B, num_tokens, _, _ = x.shape
            cur_pe = self.pe.unsqueeze(2).repeat(B, 1, 1, 1)[:, :num_tokens]
        else: 
            raise ValueError(f"Wrong shape {x.shape = }. It must have 3 or 4 dims...")
            
        y = x + cur_pe
        y = self.dropout(y)
        return y



class MLP(nn.Module):
    """
    2-Layer Multi-Layer Perceptron used in transformer blocks
    
    Args:
    -----
    in_dim: int
        Dimensionality of the input embeddings to the MLP
    hidden_dim: int
        Hidden dimensionality of the MLP
    out_dim: int or None
        Output dimensionality of the MLP.
        If not given (i.e. None), it is the same as the input dimensionality
    use_gelu: bool
        If True, the GELU activation function is used. Otherwise we use ReLU
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim=None, use_gelu=True):
        """ MLP Initializer """
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        activation = nn.ReLU() if not use_gelu else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, x):
        """ Forward """
        y = self.mlp(x)
        return y

