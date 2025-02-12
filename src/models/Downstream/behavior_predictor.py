""" 
Models used to perform behavior cloning from latent actions
"""

import torch
import torch.nn as nn

from models.BlocksUtils.attention import TransformerEncoderBlock



class BehaviorCloner(nn.Module):
    """ 
    Base class from which all Behavior Cloning modules inherit
    """
    
    def __init__(self, slot_dim, emb_dim, action_dim):
        """ Module initialzier """
        super().__init__()
        self.slot_dim = slot_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim
        
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, emb_dim)
        )        
        self.latent_action_predictor = nn.Linear(emb_dim, action_dim)
        return

    def forward(self, *_, **__):
        """ Forward pass """
        raise NotImplementedError("Base 'BehaviorCloner' does not implement forward")


    @staticmethod
    def build_slot_causal_mask(seq_len, num_slots, device):
        """
        Obtaining a binary masking attiontion pattern for the 
        slot-based transformer.
        The mask is diagonal looking, but with a 'staircase' pattern.
        Additionally, we also mask the previous ACTION tokens.
        """
        num_tokens = seq_len * num_slots
        mask = torch.zeros((num_tokens, num_tokens), device=device)        
        for i in range(seq_len):
            mask[num_slots*i:, num_slots*i:num_slots*(i + 1)] = 1.
            mask[num_slots * (i+1):, num_slots*i] = 0  # mask previous ACTION tokens
        return mask



class MarkovBehaviorCloner(BehaviorCloner):
    """ 
    Simple behavior cloning module that predicts a Latent Embedding
    given the latest observation, assuming a Markov Decission Process.
    """
    
    def __init__(self, slot_dim, emb_dim, num_layers, num_heads, head_dim,
                 mlp_dim, action_dim):
        """ Module initializer """
        super().__init__(
                slot_dim=slot_dim,
                emb_dim=emb_dim,
                action_dim=action_dim,
            )       

        # transformer to process the slot differences.
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

    def forward(self, slots):
        """
        Given one or more sets of object slots from a sequence, predicting the actions that 
        perform the desired behavior.
        
        Args:
        -----
        slots: torch tensor
            Object slots encoded from the input frames.
            Shape is (B, N_frames, N_slots, slot_dim)
            
        Returns:
        --------
        action_pred: torch tensor
            Sampled action vectors. Shape is (B, num_frames, num_slots, slot_dim)
        """
        assert len(slots.shape) == 4, f"{slots.shape = } must be (B, N, num_slots, slot_dim)"
        B, N, num_slots, _ = slots.shape
        
        # embedding slots and appending ACTION token
        slot_embs = self.slot_encoder(slots.flatten(0, 2)).reshape(B, N, num_slots, -1)
        act_token = self.act_token.repeat(B, N, 1, 1)
        all_tokens = torch.cat([act_token, slot_embs], dim=2)
        
        # reshaping time-steps into and aggregating information into ACTION token
        all_tokens = all_tokens.reshape(B * N, num_slots + 1, -1)
        output_token = self.transformer(all_tokens)[:, 0]

        # predicting actions
        action_pred = self.latent_action_predictor(output_token)
        action_pred = action_pred.reshape(B, N, self.action_dim)
       
        return action_pred
   
   
   

    
