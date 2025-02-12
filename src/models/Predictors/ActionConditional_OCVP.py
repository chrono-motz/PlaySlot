""" 
Implementation of Action-Conditional OCVP Model.

This models perform object-centric video prediction by conditioning
the forecasting on the ground-truth robot controls/actions.
"""

import torch.nn as nn

from models.Predictors.DynamicsModels import \
        SlotGPTDymamicsModel, \
        MarkovTransformerDynamicsModel



class ActionCondtionalOCVP(nn.Module):
    """ 
    Action-Conditional OCVP Models.

    This model perform object-centric video prediction by conditioning
    the forecasting on robot controls/actions.
    
    Args:
    -----
    slot_dim: int
        Slot dimensionality
    raw_action_dim: int
        Dimensionality of the robot actions/controls used for conditioning
        the precition process
    action_embed_dim: int
        Dimensionality to which the actions will be embedded
    condition_mode: string
        Defines how to condition the predictions:
            - 'sum': projecting the actions and adding to the embedded slots.
            - 'concat': concatenating the action embeddings to the embedded slots
    autoregressive_dynamics: bool
        If True, dynamics model is learned in an autoregressive manner.
        Otherwise, it is learned in parallel via teacher forcing.
    DynamicsModel: dict
        Parameters for the Dynamics Model
    """
    
    
    def __init__(self, slot_dim, raw_action_dim, action_embed_dim, condition_mode,
                 autoregressive_dynamics, DynamicsModel):
        """ Module initializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.raw_action_dim = raw_action_dim
        self.action_embed_dim = action_embed_dim
        self.condition_mode = condition_mode
        self.autoregressive_dynamics = autoregressive_dynamics

        self.action_encoder = nn.Sequential(
                nn.Linear(raw_action_dim, action_embed_dim)
            )
        self.dynamics_model = self._get_dynamics_model(
                dynamics_model_params=DynamicsModel
            )
        return
    
    
    def _get_dynamics_model(self, dynamics_model_params):
        """ Instanciating the dynamics module """        

        if "model_name" not in dynamics_model_params.keys():
            raise KeyError(f"DynamicsModel params do not have 'model_name'...")

        module_name = dynamics_model_params.get("model_name", None)
        module_params = dynamics_model_params.get("model_params", None)

        if module_name == "MarkovTransformerDynamicsModel":
            dynamics_model = MarkovTransformerDynamicsModel(
                    slot_dim=self.slot_dim,
                    num_actions=4,  # needed only for legacy
                    action_dim=self.action_embed_dim,
                    condition_mode=self.condition_mode,
                    use_variability=False,
                    action_is_one_hot=False,
                    **module_params
                )
        elif module_name == "SlotGPTDymamicsModel":
            dynamics_model = SlotGPTDymamicsModel(
                    slot_dim=self.slot_dim,
                    num_actions=4,  # needed only for legacy
                    action_dim=self.action_embed_dim,
                    condition_mode=self.condition_mode,
                    use_variability=False,
                    action_is_one_hot=False,
                    **module_params
                )
        else:
            raise NameError(f"Upsi, Dynamics module name {module_name} not recognized...")
        return dynamics_model


    def forward(self, slot_history, actions, num_seed, num_preds, **kwargs):
        """ Forward pass through model in training mode """ 
        # making sure we have a correct number of actions and frames
        num_frames, num_slots = slot_history.shape[1], slot_history.shape[2]
        num_actions = actions.shape[1]
        if num_frames < (num_seed + num_preds - 1):
            raise ValueError(f"{num_frames = } too small for {(num_seed, num_preds) = }")
        if num_actions < (num_seed + num_preds - 1):
            raise ValueError(f"{num_actions = } too small for {(num_seed, num_preds) = }")
        slot_history = slot_history[:, :num_seed + num_preds - 1]
        actions = actions[:, :num_seed + num_preds - 1]
        
        # embedding actions and repeating per-slot
        B, num_pred_steps = actions.shape[0], actions.shape[1] 
        action_embs = self.action_encoder(actions.flatten(0, 1))
        action_embs = action_embs.reshape(B, num_pred_steps, 1, -1)
        action_embs = action_embs.repeat(1, 1, num_slots, 1) 

        # prediction of future slots condition on past slots and actions
        pred_slots = self.dynamics_model(
                slots=slot_history,
                action_protos=action_embs,
                action_vars=None,
                num_seed=num_seed,
                num_preds=num_preds,
                autoregressive=self.autoregressive_dynamics
            )

        model_out = {
            "pred_slots": pred_slots,
        }        
        return pred_slots, model_out