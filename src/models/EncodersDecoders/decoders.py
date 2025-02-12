"""
Factory of decoder modules

Supported decoders are:
    - Decoder: Simple convolutional decoding cascading Conv + BN + Act. + (upsample)
"""

import torch.nn as nn
import torch.nn.functional as F

from lib.logger import print_
from models.BlocksUtils.model_blocks import ConvBlock


DECODERS = ["ConvDecoder"]


def get_decoder(in_channels, decoder, **kwargs):
    """ Instanciating a decoder given the model name and parameters """
    decoder_name = decoder["decoder_name"]
    decoder_params = decoder["decoder_params"]

    if decoder_name not in DECODERS:
        raise ValueError(f"Unknwon decoder_name {decoder_name}. Use one of {DECODERS}")

    if(decoder_name == "ConvDecoder"):
        decoder = Decoder(
            in_channels=in_channels,
            hidden_dims=decoder_params.pop("num_channels"),
            kernel_size=decoder_params.pop("kernel_size"),
            upsample=decoder_params.pop("upsample"),
            out_channels=kwargs.get("out_channels", 4),
            **decoder_params
        )

    print_("Decoder:")
    print_(f"  --> Decoder={decoder_name}")
    print_(f"  --> in_channels={in_channels}")
    for k, v in kwargs.items():
        print_(f"  --> {k}={v}")
    return decoder



class Decoder(nn.Module):
    """
    Simple fully convolutional decoder

    Args:
    -----
    in_channels: int
        Number of input channels to the decoder
    hidden_dims: list
        List with the hidden dimensions in the decoder. Final value is the number of output channels
    kernel_size: int
        Kernel size for the convolutional layers
    upsample: int or None
        If not None, feature maps are upsampled by this amount after every hidden convolutional layer
    """

    def __init__(self, in_channels, hidden_dims, kernel_size=5, upsample=None,
                 out_channels=4, **kwargs):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[0]
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.upsample = upsample
        self.out_channels = out_channels

        self.decoder = self._build_decoder()
        return

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.hidden_dims[i]
            modules.append(block)
            if isinstance(self.upsample, int) and self.upsample is not None and i > 0:
                modules.append(Upsample(scale_factor=self.upsample))
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_channels,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, x):
        """ Forward pass of the decoder """
        y = self.decoder(x)
        return y



class Upsample(nn.Module):
    """
    Overriding the upsample class to avoid an error of nn.Upsample with large tensors
    """

    def __init__(self, scale_factor):
        """ Module initializer """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ Forward pass """
        y = F.interpolate(x.contiguous(), scale_factor=self.scale_factor, mode='nearest')
        return y

    def __repr__(self):
        """ Nice printing """
        str = f"Upsample(scale_factor={self.scale_factor})"
        return str
    

