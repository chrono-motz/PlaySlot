""" 
Latent action predictor modules from PlaySlot.

Given a pretrained quatizer SAVi object-centric decomposition module, this model:
  - Computes latent actions that explain the transition between consecutive frames
  - Predicts future slots conditioned on past states and latent actions.  
"""

import torch
import torch.nn as nn

from models.Predictors.DynamicsModels import \
        SlotGPTDymamicsModel, \
        MarkovTransformerDynamicsModel
from models.Predictors.LatentAction import \
        VQSingleSlotLatentAction, \
        VQSimpleMLPSlotLatentAction


__all__ = [
        "SlotLatentPredictor",         # PlaySlot module with InvDynM
        "SlotSingleActionPredictor"    # PlaySlot module with InvDynS
    ]



class BaseSlotLatentAction(nn.Module):
    """
    Base Slot Latent-Action module from which all variants inherit
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots.
    num_actions: int
        Number of possible actions
    action_dim: int
        Dimensionality of the action embeddings
    condition_mode: string
        Mode used for conditioning the slots. Can be ['sum', 'concat']
    use_variability: bool
        If True, a variability embedding is used
    autoregressive_dynamics: bool
        If True, dynamics model is learned in an autoregressive manner.
        Otherwise, it is learned in parallel via teacher forcing.
    """

    def __init__(self, slot_dim, num_actions, action_dim, condition_mode,
                 use_variability, autoregressive_dynamics):
        """ Module initializer """
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.slot_dim = slot_dim
        self.condition_mode = condition_mode
        self.use_variability = use_variability
        self.autoregressive_dynamics = autoregressive_dynamics
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
                    num_actions=self.num_actions,
                    action_dim=self.action_dim,
                    condition_mode=self.condition_mode,
                    use_variability=self.use_variability,
                    **module_params
                )
        elif module_name == "SlotGPTDymamicsModel":
            dynamics_model = SlotGPTDymamicsModel(
                    slot_dim=self.slot_dim,
                    num_actions=self.num_actions,
                    action_dim=self.action_dim,
                    condition_mode=self.condition_mode,
                    use_variability=self.use_variability,
                    **module_params
                )
        else:
            raise NameError(f"Upsi, Dynamics module name {module_name} not recognized...")
        return dynamics_model


    @torch.no_grad()
    def forward_multiple_samples(self, slot_history, num_samples, num_seed, num_preds,
                                 use_posterior=False):
        """ 
        Computing a forward pass through the model for N different random samples, thus
        generating N different future video forecasts
        
        Args:
        -----
        slot_history: torch tensor
            Slots from the encoded video frames.
            Shape is (B, num_seed + num_preds, num_objs, slot_dim)
        num_samples: int
            Number of random samples to use to generate future forecasts.
            If use_posterior=True, the first sample will correspond to the posterior.
        num_seed: int
            Number of seed frames
        num_preds: int
            Number of frames to predict autoregressively for
        use_posterior: bool
            If True, the posterior actions are also used for prediction
            
        Returns:
        --------
        all_pred_slots: torch Tensor
            Predictions using N different random samples.
            Shape is (B, num_samples, num_preds, num_objs, slot_dim)
        """
        # making sure we have slots for the required number of frames
        num_frames = slot_history.shape[1]
        num_slots = slot_history.shape[2]
        seed_slot_history = slot_history[:, :num_seed]
        if use_posterior and num_frames < num_preds + num_seed:
            raise ValueError(
                f"If {use_posterior = }, {num_frames = } must be >= {num_seed + num_preds}"
            )
        elif use_posterior is False and num_frames < num_seed:
            raise ValueError(f"{num_frames = } must be >= {num_seed = }")

        all_pred_slots, all_actions = [], []
        
        # inferring latent actions with InvDyn and using them for generation
        if use_posterior:
            out_dict = self.latent_action.compute_actions(slot_history)
            action_protos = out_dict.get("action_proto")
            action_vars = out_dict.get("action_variability") if self.use_variability else None
            if len(action_protos.shape) == 3:  # single action model
                action_protos = action_protos.unsqueeze(2).repeat(1, 1, num_slots, 1)
            cur_pred_slots, _ = self.autoregressive_inference(
                    seed_slots=seed_slot_history,
                    action_protos=action_protos,
                    action_vars=action_vars,
                    N=num_preds
                )
            all_pred_slots.append(cur_pred_slots)
            
        # prior samples: generation using random actions and variability embeddings
        num_samples_prior = num_samples - 1 if use_posterior else num_samples
        for _ in range(num_samples_prior):
            cur_pred_slots, cur_actions = self.autoregressive_inference(
                    seed_slots=seed_slot_history,
                    action_protos=None,
                    action_vars=None,
                    N=num_preds
                )
            all_pred_slots.append(cur_pred_slots)
            all_actions.append(cur_actions)

        # aggregating all predictions
        all_pred_slots = torch.stack(all_pred_slots, dim=1)
        if len(all_actions) > 0:
            all_actions = torch.stack(all_actions, dim=1)
        return all_pred_slots, all_actions


    @torch.no_grad()
    def autoregressive_inference(self, seed_slots, action_protos=None,
                                 action_vars=None, N=10):
        """ 
        Predicting the object slots from the next N time-steps
        in an autoregressive manner
        
        Args:
        -----
        seed_slots: torch tensor
            Object slots from the seed frames.
            Shape is (B, num_frames, num_slots, slot_dim)
        action_protos: torch tensor or None
            If provided, these action prototypes are used autoregressive prediction of
            future video frames.
            Otherwise, random actions are sampled from the actions codebook for prediction.
        action_vars: torch tensor or None
            If provided, these vectors are used as variability embeddings for forecasting.
            Otherwise, new variability embeddings will be randomly generated.
        N: int
            Number of time-steps to predict autoregressively for
        """
        (B, seed_frames, num_slots, slot_dim) = seed_slots.shape

        # sampling random actions or using given action embeddings
        if action_protos is None:
            action_protos, action_idx = self.get_random_actions(
                    slots=seed_slots,
                    num_preds=N
                )
        else:
            action_idx = None
            if len(action_protos.shape) != 4:  # inflating for Single-Action model
                action_protos = action_protos.unsqueeze(-2).repeat(1, 1, num_slots, 1)
        
        # sampling random variability embeddings or using the given ones
        if not self.use_variability:
            action_vars = None
        elif action_vars is None:
            assert action_idx is not None
            action_vars = self.get_random_latents(
                    slots=seed_slots,
                    num_preds=N,
                    src_action_idx=action_idx
                )
        else:
            if len(action_vars.shape) != 4:  # inflating for Single-Action model
                action_vars = action_vars.unsqueeze(-2).repeat(1, 1, num_slots, 1)
        
        # autoregressive prediction
        all_pred_slots = self.dynamics_model(
                slots=seed_slots,
                action_protos=action_protos,
                action_vars=action_vars,
                num_seed=seed_frames,
                num_preds=N,
                autoregressive=True,
                get_pred_only=True
            )
        all_pred_slots = all_pred_slots.reshape(B, N, num_slots, slot_dim)
        return all_pred_slots, action_idx
    
    
    def get_actions_emb(self, action_idx, size=(1,)):
        """ Fetching action embeddings given action idx """
        convert_shape = None
        if isinstance(action_idx, int):
            action_idx = [action_idx]
        elif torch.is_tensor(action_idx):
            convert_shape = action_idx.shape
            action_idx = action_idx.flatten().tolist()
        if not isinstance(action_idx, (list, tuple)):
            raise TypeError(f"{type(action_idx) = } must be a list/tuple...")
            
        action_emb = torch.stack([self.latent_action.get_action(
                action_idx=idx,
                shape=size
            )[0] for idx in action_idx])

        if convert_shape is not None:
            action_emb = action_emb.reshape(*convert_shape, -1)
        return action_emb




class SlotLatentPredictor(BaseSlotLatentAction):
    """
    Latent action predictor model that predicts the subsequent Object Slots
    conditioned on unsupervised action vectors.
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots.
    num_actions: int
        Number of possible actions
    action_dim: int
        Dimensionality of the action embeddings
    condition_mode: string
        Method employed for action conditioning. This can be ['sum', 'concat']
    use_variability: bool
        If True, a variability embedding is used in order to add some continuous 
        stochasticity in addition to the discrete actions.
        Otherwise, only the discrete actions will explain the dynamics.
    autoregressive_dynamics: bool
        If True, dynamics model is learned in an autoregressive manner.
        Otherwise, it is learned in parallel via teacher forcing.
    LatentAction: dict
        Parameters use to instanciate the LatentAction module
    DynamicsModel: dict
        Parameters use to instanciate the DynamicsModel module
    """

    def __init__(self, slot_dim, num_actions, action_dim, condition_mode, use_variability,
                 autoregressive_dynamics, LatentAction, DynamicsModel):
        """ Module initializer """
        super().__init__(
                slot_dim=slot_dim,
                num_actions=num_actions,
                action_dim=action_dim,
                use_variability=use_variability,
                condition_mode=condition_mode,
                autoregressive_dynamics=autoregressive_dynamics
            )                
        self.latent_action = self._get_latent_action(
                latent_action_params=LatentAction
            )
        self.dynamics_model = self._get_dynamics_model(
                dynamics_model_params=DynamicsModel
            )
        return
        
    def _get_latent_action(self, latent_action_params):
        """ Instanciating latent action model """
        if "model_name" not in latent_action_params.keys():
            raise KeyError(f"LatentAction params do not have 'model_name'...")
        module_name = latent_action_params.get("model_name", None)
        module_params = latent_action_params.get("model_params", None)
        
        if module_name == "VQSimpleMLPSlotLatentAction":
            latent_action_model = VQSimpleMLPSlotLatentAction(
                slot_dim=self.slot_dim,
                num_actions=self.num_actions,
                action_dim=self.action_dim,
                **module_params
            )
        else:
            raise NameError(f"Latent Action model {module_name} is not supported...")
        return latent_action_model


    def forward(self, slots, num_seed, num_preds, **kwargs):
        """
        Forward pass through model in training mode:
            - First, we compute the action prototypes and variability embeddings
              that encode dynamics between consecutive time-steps
            - Then we predict the future slots using the past slots and the inferred
              action prototypes and variability embeddings.
        
        Args:
        -----
        slots: torch tensor
            Object slots from the past frames as well as from the subsequent frame.
            Shape is (B, num_frames + 1, num_slots, slot_dim)

        Returns:
        --------
        pred_slots: torch tensor
            Predicted subsequent object slots.
            Shape is (B, num_frames, num_slots, slot_dim)
        model_out: dict
            Dictionary containing VQ losses, as well as other information about actions.
        """
        num_frames = slots.shape[1]
        if num_frames < num_seed + num_preds:
            raise ValueError(f"{num_frames = } must be >= {num_seed + num_preds = }")
        if num_frames > num_seed + num_preds:
            slots = slots[:, :num_seed + num_preds]
  
        # computing posterior actions and variability embs. with Latent Action model
        latent_action_out = self.latent_action(slots)
        action_protos = latent_action_out.pop("action_proto")
        action_vars = latent_action_out.get("action_variability", None)

        slots = slots[:, :-1]  # removing last image, which was only needed for the action
        
        # autoregressive decoding via the Dynamcis Model
        pred_slots = self.dynamics_model(
                slots=slots,
                action_protos=action_protos,
                action_vars=action_vars if self.use_variability else None,
                num_seed=num_seed,
                num_preds=num_preds,
                autoregressive=self.autoregressive_dynamics
            )

        model_out = {
            "pred_slots": pred_slots,
            **latent_action_out
        }        
        return pred_slots, model_out

    def get_random_actions(self, slots, num_preds):
        """ Sampling random actions """
        B, seed_frames, num_slots, _ = slots.shape
        action_embs, action_idx = self.latent_action.get_action(
                    shape=(B, seed_frames + num_preds, num_slots)
                )
        return action_embs, action_idx

    def get_random_latents(self, slots, num_preds, src_action_idx=0):
        """ Sampling random variability latents """
        B, seed_frames, num_slots, _ = slots.shape
        size = (B, seed_frames+num_preds, num_slots)
        
        # idxs of target actions
        target_action_idxs = torch.randint(
                low=0,
                high=self.num_actions,
                size=size
            ).flatten()

        # computing embeddings and sampling an interpolation, which will define
        # the action variability latent vector
        source_emb = self.get_actions_emb(action_idx=src_action_idx, size=(1,))
        source_emb = source_emb.reshape(*size, -1)
        target_embs = self.get_actions_emb(action_idx=target_action_idxs, size=(1,))
        target_embs = target_embs.reshape(*size, -1)
    
        alpha = torch.rand(size=size, device=slots.device) / 2   # rand. in range [0, 0.5]
        latents = (target_embs - source_emb) * alpha.unsqueeze(-1)
        return latents    




class SlotSingleActionPredictor(BaseSlotLatentAction):
    """
    Latent action predictor model that predicts the subsequent Object Slots conditioned
    on a SINGLE unsupervised action vector, instead of one vector per slot.
    
    Args:
    -----
    slot_dim: int
        Dimensionality of the object slots.
    num_actions: int
        Number of possible actions
    action_dim: int
        Dimensionality of the action embeddings
    condition_mode: string
        Method used for action conditioning. Options are ['sum', 'concat']
    use_variability: bool
        If True, a variability embedding is computed for further stochasticity
    autoregressive_dynamics: bool
        If True, dynamics model is learned in an autoregressive manner.
        Otherwise, it is learned in parallel via teacher forcing.
    LatentAction: dict
        Parameters use to instanciate the LatentAction module
    DynamicsModel: dict
        Parameters use to instanciate the DynamicsModel module
    """
    
    CONDITION_MODES = ["sum", "concat"]

    
    def __init__(self, slot_dim, num_actions, action_dim, condition_mode,
                 use_variability, autoregressive_dynamics, LatentAction, DynamicsModel):
        """ Module initializer """
        super().__init__(
                slot_dim=slot_dim,
                num_actions=num_actions,
                action_dim=action_dim,
                condition_mode=condition_mode,
                use_variability=use_variability,
                autoregressive_dynamics=autoregressive_dynamics
            )                
        self.latent_action = self._get_latent_action(
                latent_action_params=LatentAction
            )
        self.dynamics_model = self._get_dynamics_model(
                dynamics_model_params=DynamicsModel
            )
        return
    
    def _get_latent_action(self, latent_action_params):
        """ Instanciating latent action model """
        if "model_name" not in latent_action_params.keys():
            raise KeyError(f"LatentAction params do not have 'model_name'...")
        module_name = latent_action_params.get("model_name", None)
        module_params = latent_action_params.get("model_params", None)
        
        if module_name == "VQSingleSlotLatentAction":
            latent_action_model = VQSingleSlotLatentAction(
                slot_dim=self.slot_dim,
                num_actions=self.num_actions,
                action_dim=self.action_dim,
                **module_params
            )
        else:
            raise NameError(f"Latent Action model {module_name} is not supported...")
        
        return latent_action_model

        
    def forward(self, slots, num_seed, num_preds, **kwargs):
        """
        Forward pass through model
        """ 
        latent_action_out = self.latent_action(slots)
        action_protos = latent_action_out.pop("action_proto")
        action_vars = latent_action_out.get("action_variability", None)
        slots = slots[:, :-1]  # removing last image, which was only needed for the action

        # repeat action per slot
        num_slots = slots.shape[2]
        action_protos = action_protos.unsqueeze(dim=2).repeat(1, 1, num_slots, 1) 
        if action_vars is not None:
            action_vars = action_vars.unsqueeze(dim=2).repeat(1, 1, num_slots, 1) 
        pred_slots = self.dynamics_model(
                slots=slots,
                action_protos=action_protos,
                action_vars=action_vars if self.use_variability else None,
                num_seed=num_seed,
                num_preds=num_preds,
                autoregressive=self.autoregressive_dynamics
            )

        model_out = {
            "pred_slots": pred_slots,
            **latent_action_out
        }        
        return pred_slots, model_out


    def get_random_actions(self, slots, num_preds):
        """ Sampling random actions """
        B, seed_frames, num_slots, _ = slots.shape
        action_embs, action_idx = self.latent_action.get_action(
                shape=(B, seed_frames + num_preds)
            )
        action_embs = action_embs.unsqueeze(2).repeat(1, 1, num_slots, 1) 
        return action_embs, action_idx


    def get_random_latents(self, slots, num_preds, src_action_idx=0):
        """ Sampling random variability latents """
        B, seed_frames, num_slots, _ = slots.shape
        size = (B, seed_frames+num_preds, 1)
        
        # idxs of target actions
        target_action_idxs = torch.randint(
                low=0,
                high=self.num_actions,
                size=size
            ).flatten()
        
        # computing embeddings and sampling an interpolation, which will define
        # the action variability latent vector
        source_emb = self.get_actions_emb(action_idx=src_action_idx, size=(1,))
        source_emb = source_emb.reshape(*size, -1)
        target_embs = self.get_actions_emb(action_idx=target_action_idxs, size=(1,))
        target_embs = target_embs.reshape(*size, -1)
        
        alpha = torch.rand(size=size, device=slots.device) / 2   # rand. in range [0, 0.5]
        latents = (target_embs - source_emb) * alpha.unsqueeze(-1)
        return latents




#
