"""
Implementation of predictor modules and wrapper functionalities
"""

import torch
import torch.nn as nn
from lib.logger import print_


__all__ = ["PredictorWrapper"]


class PredictorWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies a predictor on a sequence of data.
    This module is used for Transformer and OCVP predictors

    Args:
    -----
    exp_params: dict
        Dictionary containing the experiment parameters
    predictor: nn.Module
        Instanciated predictor module to wrap.
    """

    def __init__(self, exp_params, predictor):
        """ Module initializer """
        super().__init__()
        self.exp_params = exp_params
        self.predictor = predictor

        # prediction training and inference parameters
        self.num_context = exp_params["prediction_params"]["num_context"]
        self.num_preds = exp_params["prediction_params"]["num_preds"]
        self.teacher_force = exp_params["prediction_params"]["teacher_force"]
        self.video_length = exp_params["prediction_params"]["sample_length"]
        self.input_buffer_size = exp_params["prediction_params"]["input_buffer_size"]
        
        self.predictor_name = exp_params["predictor"]["predictor_name"]
        self.predictor_params = exp_params["predictor"]["predictor_params"]
        self._set_buffer_size()
        return

    def forward(self, slot_history, get_pred_only=False, **kwargs):
        """
        Forward pass through any Transformer-based predictor module

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)
        get_pred_only: bool
            If True, only 'predicted frames' (i.e. after context) are computed.
            Otherwise, all predictions are computed, i.e., 1:num_context + num_preds

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        self._is_teacher_force()
        start_frame = self.num_context if get_pred_only else 1
        num_preds = self.num_preds if get_pred_only else self.num_context + self.num_preds - 1
        
        predictor_input = slot_history[:, :start_frame].clone()  # inial token buffer

        pred_slots = []
        for t in range(num_preds):
            cur_preds = self.predictor(predictor_input)[:, -1]  # get current pred sltos
            next_input = slot_history[:, start_frame + t] if self.teacher_force else cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots, None

    def _is_teacher_force(self):
        """
        Updating the teacher force value, depending on the training stage
            - In eval-mode, then teacher-forcing is always false
            - In train-mode, then teacher-forcing depends on the predictor parameters
        """
        if not self._is_in_train_mode():
            self.teacher_force = False
        else:
            self.teacher_force = self.exp_params["prediction_params"]["teacher_force"]
        return

    def _is_in_train_mode(self):
        """ Checks if the predictor is in train mode or not """
        return self.predictor.training

    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.

        We keep a moving window over the input tokens, dropping the oldest slots
        if the buffer size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs
    
    def _set_buffer_size(self):
        """
        Setting the buffer size given the predicton parameters
        """
        if self.input_buffer_size is None:
            print_(f""" --> {self.predictor_name} buffer size is 'None'.
                   Setting it as {self.num_context}""")
            self.input_buffer_size = self.num_context
        if self.input_buffer_size < self.num_context:
            print_(f"  --> {self.predictor_name}'s {self.input_buffer_size = } is too small.")
            print_(f"  --> Using {self.num_context} instead...")
            self.input_buffer_size = self.num_context
        else:
            print_(f"  --> Using buffer size {self.input_buffer_size}...")
            
        return


