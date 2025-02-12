""" 
Inverse Dynamics modules that estimate Latent Actions given
consecutive pairs of observations.

These modules take a sequence of observations [x1, ...., xN] and compute the 
actions a_{t-1}^t that generate observation x_t from x_{t-1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BlocksUtils.attention import TransformerEncoderBlock
from models.BlocksUtils.quantization import VectorQuantizer, EmaVectorQuantizer


__all__ = [
    "VQSingleSlotLatentAction",     # InvDynS: single-action slot-based
    "VQSimpleMLPSlotLatentAction",  # InvDynM: multi-action slot-based
]



class BaseSlotLatentAction(nn.Module):
    """ 
    Base module for InverseDynamics modules that learn latent actions based on
    slots from consecutive frames.
    This module abstracts some of the common functionality.
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots
    emb_dim: int
        Token dimensionality in the InvDyn module
    action_dim: int
        Dimensionality of the latent actions
    num_actions: int
        Number of action prototypes to use
    use_ema_vq: bool
        If True, the VQ codebook of action prototypes is updated via an EMA.
        Otherwise, it is updated via backpropagation with the copy-gradients trick.
    """
    
    def __init__(self, slot_dim, emb_dim, action_dim, num_actions,
                 use_ema_vq=False,**kwargs):
        """ Module initializer """
        super().__init__()        
        self.slot_dim = slot_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.use_ema_vq = use_ema_vq
        
        # quantizazion and action prediction
        self.mean_fc = nn.Linear(emb_dim, action_dim)
        self.variance_fc = nn.Linear(emb_dim, action_dim)
        
        # codebook of action prototypes
        Quantizer = EmaVectorQuantizer if use_ema_vq else VectorQuantizer
        self.quantizer = Quantizer(
                num_embs=num_actions,
                emb_dim=action_dim,
                nbooks=1,
            )

        self._setup_slot_encoder()
        return
    
        
    def sample(self, mean, var, eps=1e-6):
        """ Sampling a latent vector from a Gaussian distribution """
        noise = torch.randn(mean.size(), dtype=torch.float32).to(mean.device)
        z = noise * torch.sqrt(var + eps) + mean
        return z
    

    def comput_action_dist(self, tokens):
        """
        Computing dynamics and action distributions, parameterized by their mean
        and standard deviation, given the predicted action tokens 
        """
        mean_token = self.mean_fc(tokens)
        var_token = torch.abs(self.variance_fc(tokens))
        action_dir_mean = mean_token[:, 1:] - mean_token[:, :-1]
        action_dir_var = var_token[:, 1:] + var_token[:, :-1]
        return action_dir_mean, action_dir_var


    def compute_actions(self, slots):
        """ Just a wraper of 'forward' in this case """
        return self(slots)


    def get_action(self, action_idx=None, shape=None):
        """
        Getting an action prototype from the codebook given the corresponding
        action idx.
        If None, action indices of shape 'shape' are randomly sampled.
        """
        assert shape is not None, f"A shape argument must be specified..."
        device = self.mean_fc.weight.device
        if action_idx is None:
            action_idx = torch.randint(
                    low=0,
                    high=self.quantizer.num_embs,
                    size=shape,
                    device=device
                )
        else:
            action_idx = torch.tensor(action_idx, device=device).expand(shape)
        action_protos = self.quantizer.get_codebook_entry(action_idx)
        return action_protos, action_idx

    def decompose_action_latent(self, action_latent):
        """
        Decomposing a latent action vector into an action prototype from the 
        codebook and a variability embedding
        """
        action_protos, _, _ = self.quantizer(action_latent)
        action_variability, _ = self.quantizer.get_variability(
                z=action_latent,
                action_embs=action_protos
            )
        return action_protos, action_variability



class VQSingleSlotLatentAction(BaseSlotLatentAction):
    """ 
    Simple Latent Action Inverse Dynamics model that infers a single latent action
    per timestep given the slot representations of the video frames from the
    current and past time-steps.
    This module corresponds to InvDynS.
    
    This module is a simple transformer encoder with [ACT] token. This [ACT] token
    aggregates  information from all objects, and is then processed with Linear layers
    and a Vector-Quantization codebook.
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots
    emb_dim: int
        Token dimensionality in the InvDyn module
    action_dim: int
        Dimensionality of the latent actions
    num_actions: int
        Number of action prototypes to use
    num_layers: int
        Number of transformer layers in the inverse dynamics module
    use_ema_vq: bool
        If True, the VQ codebook of action prototypes is updated via an EMA.
        Otherwise, it is updated via backpropagation with the copy-gradients trick.
    """

    
    def __init__(self, slot_dim, emb_dim, action_dim, num_actions,
                 num_layers, num_heads, head_dim, mlp_dim,
                 use_ema_vq=False, **kwargs):
        """ Module initializer """
        super().__init__(
                slot_dim=slot_dim,
                emb_dim=emb_dim,
                action_dim=action_dim,
                num_actions=num_actions,
                use_ema_vq=use_ema_vq
            )

        # transformer to process the slots along with an [ACT] token.
        self.act_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = nn.Sequential(*[
                TransformerEncoderBlock(
                    embed_dim=emb_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_dim,
                    project_out=True
                )
            for _ in range(num_layers)])        
        return


    def _setup_slot_encoder(self):
        """ Instanciating per-slot encoder module """
        self.slot_encoder = nn.Sequential(
                nn.LayerNorm(self.slot_dim),
                nn.Linear(self.slot_dim, self.emb_dim)
            )
        return    


    def forward(self, slots):
        """
        Given all slots from a sequence, computing the per-time-step latent actions
        that encode the dynamics between a consecutive pair of frames.
        These latent actions are parameterized via an action prototype and 
        an action variability.
        
        Args:
        -----
        slots: torch tensor
            Object slots encoded from the input frames.
            Shape is (B, num_frames, num_slots, slot_dim)
            
        Returns:
        --------
        action_embs: torch tensor
            Sampled action vectors. Shape is (B, num_frames - 1, num_slots, slot_dim)
        vq_losses: dict
            Dictionary containing the quantization and commitement loss values.
        action_idxs: torch tensor
            Indices of the codebook correcsponding to the predicted actions.
            Shape is (B, num_frames - 1, num_slots)
        """
        if len(slots.shape) != 4:
            raise ValueError(f"{slots.shape = } must be (B, N, num_slots, slot_dim)")
        B, N, num_slots, _ = slots.shape
    
        # embedding slots and aggregating information into [ACT] token with transformer
        slot_embs = self.slot_encoder(slots.flatten(0, 2)).reshape(B, N, num_slots, -1)
        act_tokens = self.act_token.repeat(B, N, 1, 1)
        all_tokens = torch.cat([act_tokens, slot_embs], dim=2)
        all_tokens = all_tokens.reshape(B * N, num_slots + 1, -1)
        output_token = self.transformer(all_tokens)[:, 0].reshape(B, N, -1)

        # computing actions via VQ        
        action_dist_mean, action_dist_var = self.comput_action_dist(output_token)
        action_dist = torch.stack([action_dist_mean, action_dist_var], dim=2)
        sampled_latent_action = self.sample(action_dist_mean, action_dist_var)
        action_proto, vq_loss, action_idxs = self.quantizer(sampled_latent_action)
        action_variability, _ = self.quantizer.get_variability(
                z=sampled_latent_action,
                action_embs=action_proto
            )

        action_proto = action_proto.reshape(B, N-1, -1)
        action_idxs = action_idxs.reshape(B, N-1, 1)
        vq_losses = {k: v.mean() for k, v in vq_loss.items()}

        out_dict = {
                "action_dist": action_dist,
                "sampled_latent_action": sampled_latent_action,
                "action_variability": action_variability,
                "action_proto": action_proto,
                "action_idxs": action_idxs,
                "vq_losses": vq_losses,
        }
        return out_dict
    



class VQSimpleMLPSlotLatentAction(BaseSlotLatentAction):
    """ 
    Inverse Dynamics module that learns slot-wise latent actions based on
    slots from two consecutive frames.
    This module corresponds to InvDynM.
    
    Given the slots [S_1, S_2, ..., S_T] encoded from a sequence of video frames,
    we sample latent actions [a_1^2, a_2^3, ..., a_{T-1}^{T}] from the action
    distribution from slots from consecutive time steps.    
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots
    emb_dim: int
        Token dimensionality in the InvDyn module
    hidden_dim: int
        Hidden dimensionality in the MLP moduuel
    action_dim: int
        Dimensionality of the latent actions
    num_actions: int
        Number of action prototypes to use
    use_ema_vq: bool
        If True, the VQ codebook of action prototypes is updated via an EMA.
        Otherwise, it is updated via backpropagation with the copy-gradients trick.
    """

    def __init__(self, slot_dim, emb_dim, hidden_dim, action_dim, num_actions,
                 use_ema_vq=False, **kwargs):
        """ Module initializer """
        self.hidden_dim = hidden_dim
        super().__init__(
                slot_dim=slot_dim,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_actions=num_actions,
                use_ema_vq=use_ema_vq,
                **kwargs
            )
        return
    

    def _setup_slot_encoder(self):
        """ Instanciating per-slot encoder module """
        self.slot_encoder = nn.Sequential(
                nn.Linear(self.slot_dim, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.emb_dim)
            )
        return    
    

    def forward(self, slots):
        """
        Given all slots from a sequence, computing the per-slot actions and 
        reconstructing future slots.
        
        Args:
        -----
        slots: torch tensor
            Object slots encoded from the input frames.
            Shape is (B, num_frames, num_slots, slot_dim)
            
        Returns:
        --------
        action_embs: torch tensor
            Sampled action vectors.
            Shape is (B, num_frames - 1, num_slots, slot_dim)
        vq_losses: dict
            Dictionary containing the quantization and commitement loss values.
        action_idxs: torch tensor
            Indices of the codebook correcsponding to the predicted actions.
            Shape is (B, num_frames - 1, num_slots)
        """
        assert len(slots.shape) == 4, f"{slots.shape = } must be (B, N, num_slots, slot_dim)"
        B, N, num_slots, _ = slots.shape
        
        # embedding slots and computing distribution of action directions
        slot_embs = self.slot_encoder(slots.flatten(0, 2)).reshape(B, N, num_slots, -1)
        action_dir_mean, action_dir_var = self.comput_action_dist(tokens=slot_embs)

        # computing actions via VQ
        action_dist = torch.stack([action_dir_mean, action_dir_var], dim=2)
        sampled_latent_action = self.sample(action_dir_mean, action_dir_var)
        action_protos, vq_loss, action_idxs = self.quantizer(sampled_latent_action)
        action_variability, _ = self.quantizer.get_variability(
                z=sampled_latent_action,
                action_embs=action_protos,
                action_idxs=action_idxs
            )

        action_protos = action_protos.reshape(B, N-1, num_slots, -1)
        action_idxs = action_idxs.reshape(B, N-1, num_slots, 1)
        vq_losses = {k: v.mean() for k, v in vq_loss.items()}

        out_dict = {
                "action_dist": action_dist,
                "sampled_latent_action": sampled_latent_action,
                "action_variability": action_variability,
                "action_proto": action_protos,
                "action_idxs": action_idxs,
                "vq_losses": vq_losses,
        }
        return out_dict
    

