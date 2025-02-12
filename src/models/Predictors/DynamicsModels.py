""" 
Dynamics modules that receive as input a sequence of tokens and action labels,
and predict the tokens of the subsequent frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BlocksUtils.attention import TransformerDecoder
from models.BlocksUtils.model_blocks import SlotPositionalEncoding
from models.BlocksUtils.model_utils import build_slot_causal_mask


__all__ = [
    "MarkovTransformerDynamicsModel",
    "SlotGPTDymamicsModel",
]





class BaseSlotDynamicsModel(nn.Module):
    """ 
    Base class for Slot Dynamics modules, which forecast future object slots
    conditioned on past slots, latent actions, and variability embeddings.
    It basically removes some of the scaffolding from different SlotDynamics variants
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots
    embed_dim: int
        Inner dimensionality of the dynamics model. All the processing happens 
        at this dimensionality
    num_actions: int
        Number of possible actions
    action_dim: int/None
        Dimensionality of the action space/embeddings.
    condition_mode: string
        Mode used for conditioning the slots with the actions and latent vectors.
        Options are ['sum', 'concat']
    use_variability: bool
        If True, variability vectors are used for conditioning the prediction.
    residual: bool
        If True, input is added to the output of the predictor
    """
    
    CONDITION_MODES = ["sum", "concat"]

    
    def __init__(self, slot_dim=64, embed_dim=128, num_actions=None, action_dim=None,
                 condition_mode="sum", use_variability=False, residual=False, **kwargs):
        """ Module initializer """
        super().__init__()
        if condition_mode not in self.CONDITION_MODES:
            raise NameError(f"{condition_mode = } not in {self.CONDITION_MODES = }...")
        self.slot_dim = slot_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.condition_mode = condition_mode
        self.use_variability = use_variability
        self.residual  = residual

        # predictor dimensionality depends on conditioning mode and representations
        if condition_mode == "sum":
            predictor_dim = embed_dim
        elif condition_mode == "concat":
            if use_variability:
                predictor_dim = embed_dim + action_dim + action_dim
            else:
                predictor_dim = embed_dim + action_dim
        self.predictor_dim = predictor_dim

        # mapping from codeword and action dim to transformer-inner dim
        self.tok_emb = nn.Linear(slot_dim, embed_dim)
        if condition_mode == "sum":
            self.action_emb = nn.Linear(action_dim, embed_dim) 
        if self.use_variability and condition_mode == "sum":
            self.variability_emb = nn.Linear(action_dim, embed_dim) 
        
        # slot regressing head
        self.regression_head = nn.Sequential(
                nn.LayerNorm(predictor_dim),
                nn.Linear(predictor_dim, slot_dim)
            )
        return
    

    def forward(self, slots, action_protos, action_vars=None, num_seed=None,
                num_preds=None, autoregressive=None, get_pred_only=False, **kwargs):
        """
        Forward pass through the dynamics module either in parallel or in 
        autoregressive mode.
            - Autoregressive mode appends its predictions as inputs for the next time step.
            - Parallel mode predicts all slots at once using teacher forcing.
        """
        if autoregressive is None:
            raise ValueError(f"{autoregressive = } must be True/False...")
        if autoregressive:
            pred_slots = self.forward_autoregressive(
                    slots=slots,
                    action_protos=action_protos,
                    action_vars=action_vars,
                    num_seed=num_seed,
                    num_preds=num_preds,
                    get_pred_only=get_pred_only
                )
        else:
            pred_slots = self.forward_single(
                    slots=slots,
                    action_protos=action_protos,
                    action_vars=action_vars
                )
        return pred_slots
    
    
    def condition_slots(self, tokens, action_protos, action_vars=None):
        """ 
        Conditioning object slots with action embeddings.
          - Mode='sum': map actions and variability to token dim and then sum them
          - Mode='concat': simply concat tokens, actions and variabilities
        """
        assert action_protos is not None, f"'action_protos' not provided..."
        # map to token dim, and sum tokens, action and variability embeddings
        if self.condition_mode == "sum":
            assert hasattr(self, "action_emb"), f"'action_emb' layer not found!"
            action_proto_embs = self.action_emb(action_protos)
            conditioned_tokens = tokens + action_proto_embs
            if self.use_variability:
                assert action_vars is not None, f"'action_vars' not provided..."
                assert hasattr(self, "variability_emb"), f"'variability_emb' not found!"
                action_vars_embs = self.variability_emb(action_vars)
                conditioned_tokens = conditioned_tokens + action_vars_embs
        # concatenate tokens, action and variability embeddings
        elif self.condition_mode == "concat":
            if self.use_variability:
                assert action_vars is not None, f"'action_vars' not provided..."
                conditioned_tokens = torch.cat(
                        [tokens, action_protos, action_vars],
                        dim=-1
                    ) 
            else:
                conditioned_tokens = torch.cat([tokens, action_protos], dim=-1) 
        else:
            raise NameError(f"{self.condition_mode = } not in {self.CONDITION_MODES = }...")
        return conditioned_tokens    



class MarkovTransformerDynamicsModel(BaseSlotDynamicsModel):
    """ 
    Transformer dynamics model that follows the Markovian constraint, i.e., 
    the slots at time-step t depend only on the slots at time step t-1 and
    the action vectors:
        - p(s_t | s_{t-1}, a_t) 
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots
    embed_dim: int
        Dimensionality of the tokens input to the transformer predictor
    num_actions: int
        Number of possible actions
    action_dim: int/None
        If given, dimensionality of the action embeddings.
        If None, no actions conditioning is performed
    condition_mode: string
        Mode used for conditioning the slots with the actions and latent vectors.
        Options are ['sum', 'concat']
    use_variability: bool   
        If True, action variability embeddings are used.
        Otherwise, only action prototypes are used.
    residual: bool
        If True, a residual connection bridges the predictor: s_t := s_t + s_{t-1}
    head_dim: int
        Dimensionality per head.
    num_heads: int
        Number of self-attention heads
    mlp_size: int
        Hidden dim in the MLPs of the transformer
    num_layers: int
        Number of transformer blocks
    """
        
    CONDITION_MODES = ["sum", "concat"]

    
    def __init__(self, slot_dim=64, embed_dim=128, num_actions=None, action_dim=None,
                 condition_mode="sum", use_variability=False, residual=False,
                 head_dim=32, num_heads=4, mlp_size=512, num_layers=4, **kwargs):
        """ Module initializer """
        super().__init__(
                slot_dim=slot_dim,
                embed_dim=embed_dim,
                num_actions=num_actions,
                action_dim=action_dim,
                condition_mode=condition_mode,
                use_variability=use_variability,
                residual=residual
            )

        # main-body of the dynamics module
        self.dynamics_model = nn.ModuleList([
                TransformerDecoder(
                    embed_dim=self.predictor_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    kv_dim=None,
                    use_cross_attn=False,
                    project_out=True
                )
            for _ in range(num_layers)]
        )
        return


    def forward_autoregressive(self, slots, action_protos, action_vars=None, num_seed=None,
                               num_preds=None, get_pred_only=False, **kwargs ):
        """
        Forward pass through the model in autoregressive mode.

        Args:
        -----
        slots: torch.Tensor
            Input slots encoding the objects in the scene.
            Shape is (B, num_seed, num_slots, slot_dim)
        action_protos: torch.Tensor   
            Action prototypes encoding the dynamics between every consecutiv time steps.
            Shape is (B, num_seed + num_preds - 1, action_dim)
        action_vars: torch.Tensor
            Action variability vectors.
            Shape is (B, num_seed + num_preds - 1, action_dim)
        num_seed: int, optional:
            Number of seed frames used in the prediction process
        num_preds: int, optional
            Number of time-steps to predict for
        get_pred_only: bool, optional
            If True, only the predicted actions will be returned.

        Returns:
        --------
        pred_slots: torch.Tensor
            Predicted object slots conditioned on past slots, action prototypes
            and variabilities.
            Shape is (B, num_seed + num_preds - 1, num_slots, slot_dim)
        """
        assert num_seed is not None, f"{num_seed = } must be provided in AR mode..."
        assert num_preds is not None, f"{num_preds = } must be provided in AR mode..."
        if self.use_variability:
            assert action_vars is not None, f"variability latents required but not provided"
        pred_slots = []
        input_slots = slots[:, :1]
        for i in range(num_seed + num_preds - 1):
            cur_pred_slots = self.forward_single(
                    slots=input_slots,
                    action_protos=action_protos[:, i:i+1],
                    action_vars=action_vars[:, i:i+1] if self.use_variability else None
                )
            if not get_pred_only or i >= num_seed - 1:
                pred_slots.append(cur_pred_slots)
            input_slots = slots[:, i+1:i+2] if i < num_seed - 1 else cur_pred_slots
        pred_slots = torch.cat(pred_slots, dim=1)
        return pred_slots


    def forward_single(self, slots, action_protos, action_vars=None, **kwargs):
        """
        Single-step forward pass through the dynamics module
        
        Args:
        -----
        tokens: torch tensor
            Object slots.
            Shape is (B, 1, n_slots, slot_dim)
        action_protos: torch tensor/None
            Action prototypes predicted.
            Shape is (B, 1, n_slots, action_dim)
        action_vars: torch tensor/None
            Continuous latent variability vectors.
            Shape is (B, 1, n_slots, action_dim)
            
        Returns:
        --------
        pred_slots: torch.Tensor
            Predicted object slots
            Shape is (B, 1, num_slots, slot_dim)
        """
        if action_protos is None:
            raise ValueError(f"'action_protos' were not provided")
        if slots.shape[:-1] != action_protos.shape[:-1]:
            raise ValueError(f"{slots.shape =} != {action_protos.shape = }")
        if self.use_variability and action_vars is None:
            raise ValueError(f"variability latents were required but not provided")
        B, num_frames, num_slots, slot_dim = slots.shape
        
        # embedding slots and conditioning with action embeddings
        token_embs = self.tok_emb(slots)
        input_embs = self.condition_slots(
                tokens=token_embs,
                action_protos=action_protos,
                action_vars=action_vars
            )

        # forward pass through transformer and regression head in order to predict next slots
        input_embs = input_embs.reshape(B * num_frames, num_slots, -1)
        for block in self.dynamics_model:
            input_embs = block(input_embs)
        pred_slots = self.regression_head(input_embs)
        pred_slots = pred_slots.reshape(B, num_frames, num_slots, slot_dim)
        if self.residual:
            pred_slots = pred_slots + slots
        return pred_slots




class SlotGPTDymamicsModel(BaseSlotDynamicsModel):
    """ 
    GPT-like model that works directly on object slots.
    It predicts the object slots at the next time step given the object slots
    from all previous time steps and actions,
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots
    action_dim: int/None
        If given, dimensionality of the action embeddings.
        If None, no actions conditioning is performed
    num_actions: int
        Number of possible actions
    condition_mode: string
        Mode used for conditioning the slots with the actions and latent vectors.
        Options are ['sum', 'concat']
    embed_dim: int
        Dimensionality of the tokens input to the transformer predictor
    head_dim: int
        Dimensionality per head.
    num_heads: int
        Number of self-attention heads
    mlp_size: int
        Hidden dim in the MLPs of the transformer
    num_layers: int
        Number of transformer blocks
    context_length: int
        Context length (in frames) of the transformer predictor
    """
    
    def __init__(self, slot_dim=64, embed_dim=128, num_actions=None, action_dim=None,
                 condition_mode="sum", use_variability=False, residual=False,
                 head_dim=32, num_heads=4, mlp_size=512, num_layers=4,
                 context_length=8, pos_enc_dropout=0.1, **kwargs):
        """ Module initializer """
        if context_length is None:
            raise ValueError(f"'SlotGPTDymamicsModel' must receive 'context_length' param...")
        super().__init__(
                slot_dim=slot_dim,
                embed_dim=embed_dim,
                num_actions=num_actions,
                action_dim=action_dim,
                condition_mode=condition_mode,
                use_variability=use_variability,
                residual=residual
        )
        self.context_length = context_length
        self.slot_causal_mask = None
        self.slots_in_causal_mask = None
        
        # positional encoding
        self.pos_emb = SlotPositionalEncoding(
                d_model=self.predictor_dim,
                max_len=context_length, 
                dropout=pos_enc_dropout
            )

        # main-body of the dynamics module
        self.dynamics_model = nn.ModuleList([
                TransformerDecoder(
                    embed_dim=self.predictor_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    kv_dim=None,
                    use_cross_attn=False,
                    project_out=True
                )
            for _ in range(num_layers)]
        )
        return


    def forward_autoregressive(self, slots, action_protos, action_vars=None,
                               num_seed=None, num_preds=None,
                               get_pred_only=False, **kwargs ):
        """
        Forward pass through the SlotGPT model in autoregressive mode.

        Args:
        -----
        slots: torch.Tensor
            Input slots encoding the objects in the scene.
            Shape is (B, num_seed, num_slots, slot_dim)
        action_protos: torch.Tensor   
            Action prototypes encoding the dynamics between every consecutiv time steps.
            Shape is (B, num_seed + num_preds - 1, action_dim)
        action_vars: torch.Tensor
            Action variability vectors.
            Shape is (B, num_seed + num_preds - 1, action_dim)
        num_seed: int, optional:
            Number of seed frames used in the prediction process
        num_preds: int, optional
            Number of time-steps to predict for
        get_pred_only: bool, optional
            If True, only the predicted actions will be returned.

        Returns:
        --------
        pred_slots: torch.Tensor
            Predicted object slots conditioned on past slots, action prototypes
            and variabilities.
            Shape is (B, num_seed + num_preds - 1, num_slots, slot_dim)
        """
        assert num_seed is not None, f"{num_seed = } must be provided in AR mode..."
        assert num_preds is not None, f"{num_preds = } must be provided in AR mode..."
        pred_slots = []
        
        num_steps_to_pred = num_preds if get_pred_only else num_seed + num_preds - 1
        input_slots = slots[:, :num_seed] if get_pred_only else slots[:, :1]
        for i in range(num_steps_to_pred):
            idx = num_seed + i if get_pred_only else i+1
            cur_action = action_protos[:, :idx]
            cur_var = action_vars[:, :idx] if action_vars is not None else None
            input_slots, cur_action, cur_var = self.enforce_window(
                    slots=input_slots,
                    action_protos=cur_action,
                    action_vars=cur_var
                )
            cur_pred_slots = self.forward_single(
                    slots=input_slots,
                    action_protos=cur_action,
                    action_vars=cur_var
                )
            cur_pred_slots = cur_pred_slots[:, -1].unsqueeze(1)
            pred_slots.append(cur_pred_slots)
            if not get_pred_only and i < num_seed - 1:
                input_slots = slots[:, :idx+1]
            else:
                input_slots = torch.cat([input_slots, cur_pred_slots], dim=1)
        pred_slots = torch.cat(pred_slots, dim=1)
        return pred_slots


    def forward_single(self, slots, action_protos, action_vars=None, causal_mask=None):
        """
        Forward pass for a single prediction step through the dynamics module
        
        Args:
        -----
        tokens: torch tensor
            Object slots. Shape is (B, num_frames, num_slots, slot_dim)
        action_protos: torch tensor/None
            Action prototypes predicted. Shape is (B, num_frames, num_slots, action_dim)
        causal_mask: torch tensor/None
            If given, causal masked employed in the predictor to enforce causality.
            If None, the causal mask is compute in here.
            
        Returns:
        --------
        pred_slots: torch tensor
            Forecasted object slots conditioned on past object slots, actions
            and variability embeddings.
            Shape is (B, num_frames, num_slots, slot_dim)
        """
        if slots.shape[:-1] != action_protos.shape[:-1]:
            raise ValueError(f"{slots.shape =} != {action_protos.shape = }")
        if self.use_variability and action_vars is None:
            raise ValueError(f"If {self.use_variability = }, latents must be provided...")
        (B, num_frames, num_slots, slot_dim), device = slots.shape, slots.device
        if num_frames > self.context_length:
            raise ValueError(f"{num_frames = } cannot be > {self.context_length = }")
        
        # embedding slots and conditioning with action embeddings
        token_embs = self.tok_emb(slots)
        input_embs = self.condition_slots(
                tokens=token_embs,
                action_protos=action_protos,
                action_vars=action_vars
            )
        
        # positional encoding
        pos_input_embs = self.pos_emb(input_embs, batch_size=B, num_slots=num_slots)
        pos_input_embs = pos_input_embs.flatten(1, 2)  # (B, n_frames * n_slots, emb_dim)
        
        # forward pass through transformer and predicting next slots
        # causal mask needed if we want to do parallel training.
        causal_mask = self.get_slot_causal_mask(
                num_frames=num_frames,
                num_slots=num_slots,
                device=device    
            )
        for block in self.dynamics_model:
            pos_input_embs = block(
                    pos_input_embs,
                    self_attn_mask=causal_mask
                )
        
        # forward pass through transformer and regression head in order to predict next slots
        pred_slots = self.regression_head(pos_input_embs)
        pred_slots = pred_slots.reshape(B, num_frames, num_slots, slot_dim)
        if self.residual:
            pred_slots = pred_slots + slots
        return pred_slots


    def enforce_window(self, slots, action_protos, action_vars=None):
        """  Enforcing that the number of frames does not exceed window-size """
        num_frames = slots.shape[1]
        if num_frames > self.context_length:
            slots = slots[:, -self.context_length:]
            action_protos = action_protos[:, -self.context_length:]
            if action_vars is not None:
                action_vars = action_vars[:, -self.context_length:]
        return slots, action_protos, action_vars


    def get_slot_causal_mask(self, num_slots, num_frames, device):
        """ Instanciating the slot causal mask or fetching it from cache """
        # If new number of slots or causal mask does not exist -->  new mask.
        if self.slot_causal_mask is None or self.slots_in_causal_mask != num_slots:
            self.slots_in_causal_mask = num_slots
            self.slot_causal_mask = build_slot_causal_mask(
                    num_slots=num_slots,
                    seq_len=self.context_length,
                    device=device
                )

        # number of frames is different --> we crop the mask
        if self.slot_causal_mask.shape[0] < num_frames * num_slots:
            raise ValueError(f"{self.slot_causal_mask.shape = } too small for #frames & #slots")
        elif self.slot_causal_mask.shape[0] > num_frames * num_slots:
            needed_size = int(num_frames * num_slots)
            slot_causal_mask = self.slot_causal_mask[:needed_size, :needed_size]
        else:
            slot_causal_mask = self.slot_causal_mask

        return slot_causal_mask
        


#

