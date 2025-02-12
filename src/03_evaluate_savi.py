"""
Evaluating a SAVI model checkpoint using image reconstruction metrics
"""

import argparse
import torch

from base.baseEvaluator import BaseEvaluator
from data import unwrap_batch_data
from lib.logger import Logger, print_, log_function, for_all_methods
import lib.utils as utils



@for_all_methods(log_function)
class Evaluator(BaseEvaluator):
    """
    Class for evaluating a SAVI model using image reconstruction metrics
    """
    
    @torch.no_grad()
    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        """
        videos, _, initializer_kwargs, _ = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        out_model = self.savi(
                videos,
                num_imgs=videos.shape[1],
                **initializer_kwargs
            )
        preds = out_model.get("recons_imgs").clamp(0, 1)
        
        # metric computation
        self.metric_tracker.accumulate(
                preds=preds.cpu().detach(),
                targets=videos.cpu().detach()
            )
        return



if __name__ == "__main__":
    utils.clear_cmd()
    
    # processing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the experiment directory",
            required=True
        )
    parser.add_argument(
            "--savi_ckpt",
            help="Checkpoint with pretrained parameters to load",
            required=True
        )
    parser.add_argument(
            "-o", "--results_name",
            help="Name to give to the results file",
            type=str, required=True
        )
    parser.add_argument(
            "--batch_size",
            help="Overriding batch size used for evaluation",
            type=int, default=0
        )
    parser.add_argument(
            "--set_expert_policy",
            help="If given, expert policy variant is used...",
            default=False, action='store_true'
        )
    args = parser.parse_args()
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    
    logger = Logger(exp_path=exp_path)
    logger.log_info(
            "Starting SAVi evaluation procedure",
            message_type="new_exp"
        )
    logger.log_arguments(args)

    print_("Initializing Evaluator...")
    evaluator = Evaluator(
            exp_path=exp_path,
            savi_ckpt=args.savi_ckpt,
            batch_size=args.batch_size,
            set_expert_policy=args.set_expert_policy,
            results_name=args.results_name
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up SAVi model and loading pretrained parameters")
    evaluator.load_savi()
    print_("Starting visual quality evaluation")
    evaluator.evaluate()


