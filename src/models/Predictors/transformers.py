"""
Implementation of predictor modules and wrapper functionalities
Code adapted from:
    - https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction
"""

import torch
import torch.nn as nn
from lib.logger import print_

from models.BlocksUtils.attention import MultiHeadSelfAttention, TransformerEncoderBlock
from models.BlocksUtils.model_blocks import SlotPositionalEncoding, MLP
from models.BlocksUtils.model_utils import init_kaiming_


__all__ = ["VanillaTransformerPredictor", "OCVPSeq", "OCVPPar"]



class VanillaTransformerPredictor(nn.Module):
    """
    Vanilla Transformer Predictor module.
    It performs self-attention over all slots in the input buffer, jointly modelling
    the relational and temporal dimensions.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, token_dim=128, hidden_dim=256,
                 num_layers=2, n_heads=4, residual=False, input_buffer_size=5,
                 pos_enc_dropout=0.1):
        """
        Module initializer
        """
        if token_dim % n_heads != 0:
            raise ValueError(f"{token_dim = } must be divisible by {n_heads = }...")
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = n_heads
        self.residual = residual
        self.pos_enc_dropout = pos_enc_dropout
        self.input_buffer_size = input_buffer_size
        print_("Instanciating Vanilla Transformer Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.num_heads}")
        print_(f"  --> pos_enc_dropout: {self.pos_enc_dropout}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # MLPs to map slot-dim into token-dim and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoderBlock(
                embed_dim=self.token_dim,
                head_dim=self.token_dim // self.num_heads,
                num_heads=self.num_heads,
                mlp_size=self.hidden_dim,
                self_attn=True,
            ) for _ in range(num_layers)]
        )

        # Custom temporal encoding. All slots from the same time step share the encoding
        self.pe = SlotPositionalEncoding(
                d_model=self.token_dim,
                max_len=input_buffer_size,
                dropout=pos_enc_dropout
            )
        
        self.attention_masks = None
        self._init_model()
        return

    def forward(self, inputs):
        """
        Foward pass through the transformer predictor module to predict
        the subsequent object slots

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps.
            Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots.
            Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # mapping slots to tokens, and applying temporal positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through transformer encoder blocks
        token_output = time_encoded_input.reshape(B, num_imgs * num_slots, self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)
        token_output = token_output.reshape(B, num_imgs, num_slots, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output

    def get_attention_masks(self):
        """
        Fetching the last computed attention maps
        """
        attn_masks = self.transformer_encoders[-1].get_attention_masks()
        return attn_masks

    @torch.no_grad()
    def _init_model(self):
        """ Parameter initialization """
        init_kaiming_(self)
        return



class OCVPSeq(nn.Module):
    """
    Sequential Object-Centric Video Prediction Transformer Module (OCVP-Seq).
    This module models the temporal dynamics and object interactions in a
    decoupled manner by sequentially applying object- and time-attention
    i.e. [time, obj, time, ...]

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is n_slots * n_imgs
    slot_dim: int
        Dimensionality of the input slots
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5, pos_enc_dropout=0.1):
        """
        Module Initialzer
        """
        if token_dim % n_heads != 0:
            raise ValueError(f"{token_dim = } must be divisible by {n_heads = }...")
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = n_heads
        self.residual = residual
        self.pos_enc_dropout = pos_enc_dropout
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating OCVP-Seq Predictor Module:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.num_heads}")
        print_(f"  --> pos_enc_dropout: {self.pos_enc_dropout}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads,
        # i.e., each head will have dim. embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPSeqLayer(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads
                ) for _ in range(num_layers)]
            )

        # custom temporal encoding.
        # All slots from the same time step share the same encoding
        self.pe = SlotPositionalEncoding(
                d_model=self.token_dim,
                max_len=input_buffer_size,
                dropout=pos_enc_dropout
            )
        
        return

    def forward(self, inputs):
        """
        Forward pass through OCVP-Seq

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps.
            Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim),
            but we only care about the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, _, num_slots, _ = inputs.shape

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through OCVP-Seq transformer blocks
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output



class OCVPSeqLayer(nn.Module):
    """
    Sequential Object-Centric Video Prediction (OCVP-Seq) Transformer Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        if token_dim % n_heads != 0:
            raise ValueError(f"{token_dim = } must be divisible by {n_heads = }...")
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_heads = n_heads
        
        self.object_encoder_block = TransformerEncoderBlock(
            embed_dim=token_dim,
            head_dim=token_dim // n_heads,
            num_heads=n_heads,
            mlp_size=hidden_dim,
            self_attn=True,
        )
        self.time_encoder_block = TransformerEncoderBlock(
            embed_dim=token_dim,
            head_dim=token_dim // n_heads,
            num_heads=n_heads,
            mlp_size=hidden_dim,
            self_attn=True,
        )
        return

    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out



class OCVPPar(nn.Module):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Predictor Module.
    This module models the temporal dynamics and object interactions in a
    dissentangled manner by applying relational- and temporal-attention in parallel.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5, pos_enc_dropout=0.1):
        """
        Module initializer
        """
        assert token_dim % n_heads == 0, f"ERROR! {token_dim = } must be divisible by {n_heads = }..."
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = n_heads
        self.pos_enc_dropout = pos_enc_dropout
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating OCVP-Par Predictor Module:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.num_heads}")
        print_(f"  --> pos_enc_dropout: {self.pos_enc_dropout}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")
        print_(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim will be split across num_heads,
        # i.e. each head will have dim embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPParLayer(
                    d_model=token_dim,
                    nhead=self.num_heads,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        self.pe = SlotPositionalEncoding(
                d_model=self.token_dim,
                max_len=input_buffer_size,
                dropout=pos_enc_dropout
            )
        return

    def forward(self, inputs):
        """
        Forward pass through Object-Centric Transformer v1

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps.
            Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim),
            but we only care about the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, _, num_slots, _ = inputs.shape

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding tokens through transformer la<ers
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # projecting back to slot-dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output



class OCVPParLayer(nn.Module):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Module.
    This module models the temporal dynamics and object interactions in a
    dissentangled manner by applying object- and time-attention in parallel.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        assert token_dim % n_heads == 0, f"ERROR! {token_dim = } must be divisible by {n_heads = }..."
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        assert n_heads >= 1

        # common layers
        self.ln_att = nn.LayerNorm(token_dim, eps=1e-6)
        self.ln_mlp = nn.LayerNorm(token_dim, eps=1e-6)
        self.mlp = MLP(
            in_dim=token_dim,
            hidden_dim=hidden_dim,
        )
        
        # attention layers
        self.object_encoder_block = MultiHeadSelfAttention(
                emb_dim=token_dim,
                head_dim=token_dim // n_heads,
                num_heads=n_heads,
                project_out=False
            )
        self.time_encoder_block = MultiHeadSelfAttention(
                emb_dim=token_dim,
                head_dim=token_dim // n_heads,
                num_heads=n_heads,
                project_out=False
            )
        return


    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-Parallel module.

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape
        inputs = self.ln_att(inputs)
        
        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        object_encoded_out = inputs.clone().reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        time_encoded_out = inputs.clone().transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
        time_encoded_out = self.time_encoder_block(time_encoded_out)
        time_encoded_out = time_encoded_out.reshape(B, num_slots, num_imgs, dim).transpose(1, 2)

        y = (object_encoded_out + time_encoded_out) / 2
        y = y + inputs
        
        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        output = z + y
        return output



