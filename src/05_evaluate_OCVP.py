"""
Evaluating an OCVP model checkpoint for video prediction.

This script can be used to evaluate the following OCVP-Models:
  - Vanilla OCVP (same as SlotFormer)
  - OCVP-Seq
  - OCVP-Par
  - Action-Conditional OCVP
"""

import argparse
import os
import torch

from base.baseEvaluator import BaseEvaluator
from data import unwrap_batch_data
from lib.config import Config
from lib.logger import Logger, print_
import lib.utils as utils



class Evaluator(BaseEvaluator):
    """
    Evaluating an OCVP model checkpoint for video prediction.
    """
    
    def __init__(self, exp_path, name_pred_exp, savi_ckpt, pred_ckpt,
                 num_seed=None, num_preds=None, batch_size=None,
                 results_name=None, set_expert_policy=False, **kwargs):
        """ Evaluator initalizer """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.pred_ckpt = pred_ckpt
        self.batch_size = batch_size
        self.results_name = results_name
        self.set_expert_policy = set_expert_policy
        
        # paths and utils
        self.plots_path = os.path.join(self.exp_path, "plots")
        self.savi_model_path = os.path.join(self.parent_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")
        self.override_num_seed_and_preds(num_seed=num_seed, num_preds=num_preds)
        return
    

    def set_evaluation(self):
        """ Instanciating modules for evaluation """
        self.metric_tracker = self.set_metric_tracker()
        self.unwrap_function = unwrap_batch_data
        return


    @torch.no_grad()
    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, actions, or metadata, among others.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        """
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]

        # fetching and preparing data
        videos, _, init_kwargs, other = self.unwrap_function(self.exp_params, batch_data)
        videos = videos.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} smaller that {num_context + num_preds = }")

        pred_imgs = self.forward_model(
                videos=videos,
                init_data=init_kwargs,
                other=other,
                num_context=num_context,
                num_preds=num_preds
            )

        # selecting predictions and targets and computing evaluation metrics
        preds_eval = pred_imgs.view(B, num_preds, C, H, W).clamp(0, 1)
        targets_eval = videos[:, num_context:num_context+num_preds, :, :].clamp(0, 1)
        self.metric_tracker.accumulate(
                preds=preds_eval.cpu().detach().clamp(0, 1),
                targets=targets_eval.cpu().detach().clamp(0, 1)
            )
        return


    def forward_model(self, videos, init_data, other, num_context, num_preds):
        """ Forward pass through the model to predict future slots"""
        B, L = videos.shape[0], videos.shape[1]

        # encoding images into object-centric slots, and temporally aligning slots
        out_model = self.savi(videos, num_imgs=L, **init_data)
        slot_history = out_model["slot_history"]

        # predicting future slots
        actions = other.get("actions").to(self.device) if "actions" in other else None
        pred_slots, _ = self.predictor(
                slot_history=slot_history,
                use_posterior=False,
                actions=actions,
                num_seed=num_context,
                num_preds=num_preds
            )
        pred_slots = pred_slots[:, num_context - 1 : num_context + num_preds - 1]

        # decoding predicted slots into predicted frames
        num_slots, slot_dim = pred_slots.shape[-2], pred_slots.shape[-1]
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        pred_imgs, _ = self.savi.decode(pred_slots_decode)
        return pred_imgs



if __name__ == "__main__":
    utils.clear_cmd()

    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the SAVi father exp. directory",
            required=True
        )
    parser.add_argument(
            "--savi_ckpt",
            help="Name of the pretrained SAVi checkpoint use",
            required=True
        )
    parser.add_argument(
            "--name_pred_exp",
            help="Name to the predictor experiment to evaluate.",
            required=True
        )
    parser.add_argument(
            "--pred_ckpt",
            help="Name of the predictor checkpoint to evaluate",
            required=True
        )
    parser.add_argument(
            "-o", "--results_name",
            help="Name to give to the results file",
            type=str, required=True
        )
    # additional arguments
    parser.add_argument(
            "--batch_size",
            help="If provided, it overrides the batch size used for evaluation",
            type=int, default=0
        )
    parser.add_argument(
            "--set_expert_policy",
            help="If given, expert policy variant is used...",
            default=False, action='store_true'
        )
    parser.add_argument(
            "--num_seed",
            help="If provided, it overrides the number of seed frames to use",
            type=int, default=None
        )
    parser.add_argument(
            "--num_preds",
            help="If provided, it overrides the number of frames to predict for",
            type=int, default=None
        )
    args = parser.parse_args()

    # sanity checks on command line arguments
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    args.savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    args.name_pred_exp = utils.process_predictor_experiment(
            exp_directory=exp_path,
            name_predictor_experiment=args.name_pred_exp,    
        )
    args.pred_ckpt = utils.process_predictor_checkpoint(
            exp_path=exp_path,
            name_predictor_experiment=args.name_pred_exp,
            checkpoint=args.pred_ckpt
        )
    if args.batch_size < 1:
        args.batch_size = None


    logger = Logger(exp_path=f"{exp_path}/{args.name_pred_exp}")
    logger.log_info(
            "Starting object-centric predictor evaluation procedure",
            message_type="new_exp"
        )
    logger.log_arguments(args)

    print_("Initializing Evaluator...")
    evaluator = Evaluator(
            exp_path=exp_path,
            name_pred_exp=args.name_pred_exp,
            savi_ckpt=args.savi_ckpt,
            pred_ckpt=args.pred_ckpt,
            num_seed=args.num_seed,
            num_preds=args.num_preds,
            batch_size=args.batch_size,
            results_name=args.results_name,
            set_expert_policy=args.set_expert_policy,
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up SAVi and predictor and loading pretrained parameters")
    evaluator.load_savi(models_path=evaluator.savi_model_path)
    evaluator.load_predictor(models_path=evaluator.models_path)

    # VIDEO PREDICTION EVALUATION
    print_("Starting video predictor evaluation")
    evaluator.set_evaluation()
    evaluator.evaluate()


