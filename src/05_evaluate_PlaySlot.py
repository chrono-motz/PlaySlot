"""
Evaluating an stochastic object-centric predictor model checkpoint.
For every sequence, we sample evaluate for it N different times using 
different random samples.
"""

import os
import argparse
import torch

from base.baseEvaluator import BaseEvaluator
from data import unwrap_batch_data
from lib.config import Config
from lib.logger import Logger, print_
import lib.utils as utils



class StochEvaluator(BaseEvaluator):
    """
    Evaluating an stochastic object-centric predictor model checkpoint.
    For every sequence, we sample evaluate for it N different times using 
    different random samples.
    """


    def __init__(self, exp_path, name_pred_exp, savi_ckpt, pred_ckpt,
                 num_seed=None, num_preds=None, batch_size=None, num_samples=10,
                 results_name=None, set_expert_policy=False, post_only=False, **kwargs):
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
        self.post_only = post_only
        
        # paths and utils
        self.plots_path = os.path.join(self.exp_path, "plots")
        self.savi_model_path = os.path.join(self.parent_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")
        self.override_num_seed_and_preds(num_seed=num_seed, num_preds=num_preds)

        # updating for 'post_only' evaluation
        if not post_only:
            self.batch_size = 1
            self.num_samples = num_samples
        else:
            self.num_samples = 1
        return


    def set_evaluation(self):
        """ Instanciating modules for evaluation """
        self.metric_tracker = self.set_metric_tracker()
        self.posterior_metric_tracker = self.set_metric_tracker()      
        self.unwrap_function = unwrap_batch_data
        return


    @torch.no_grad()
    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics
        """
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]

        # fetching and preparing data
        videos, _, init_kwargs, _ = self.unwrap_function(self.exp_params, batch_data)
        videos = videos.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(
                    f"Seq. length {L} must be >= {num_context + num_preds = }"
                )

        # predicting images
        pred_imgs = self.forward_model(
                videos=videos,
                init_data=init_kwargs,
                num_context=num_context,
                num_preds=num_preds
            )

        # selecting predictions and targets and computing evaluation metrics
        preds_eval = pred_imgs.view(B, self.num_samples, num_preds, C, H, W).clamp(0, 1)
        targets_eval = videos[:, num_context:num_context+num_preds, :, :].clamp(0, 1)

        # stochastic evaluation by keeping the best out of all prior samples
        if not self.post_only and self.num_samples > 1:
            cur_targets = targets_eval.unsqueeze(1)
            cur_targets = cur_targets.repeat(1, self.num_samples - 1, 1, 1, 1, 1)
            self.metric_tracker.accumulate(
                    preds=preds_eval[:, 1:].cpu().detach().flatten(0, 1),
                    targets=cur_targets.cpu().detach().flatten(0, 1)
                )
            self.metric_tracker.get_best_trial(num_trials=self.num_samples - 1)

        # evaluation of posterior results
        self.posterior_metric_tracker.accumulate(
                preds=preds_eval[:, 0].cpu().detach().clamp(0, 1),
                targets=targets_eval.cpu().detach().clamp(0, 1)
            )
        return
    
    
    def forward_model(self, videos, init_data, num_context, num_preds):
        """ 
        Forward pass through the PlaySlot model to autoregressively predict the frames
        """
        # encoding images into slots
        B, L = videos.shape[0], videos.shape[1]
        out_model = self.savi(videos, num_imgs=L, **init_data)
        slot_history = out_model.get("slot_history")

        # predicting slots using multiple latent actions, including inferred ones
        pred_slots, _ = self.predictor.forward_multiple_samples(
                slot_history=slot_history,
                num_samples=self.num_samples,
                num_seed=num_context,
                num_preds=num_preds,
                use_posterior=True
            )

        # decoding predicted slots into predicted frames
        num_slots, slot_dim = pred_slots.shape[-2], pred_slots.shape[-1]
        pred_slots_dec = pred_slots.reshape(
                B * self.num_samples * num_preds,
                num_slots,
                slot_dim
            )
        img_recons, _ = self.savi.decode(pred_slots_dec)

        return img_recons
    
    
    
    @torch.no_grad()
    def aggregate_and_save_metrics(self, fname=None):
        """
        Aggregating all computed metrics and saving results to logs file
        """
        if self.post_only:
            names = ["Post"]
            trackers = [self.posterior_metric_tracker]
        else:
            names = ["Post", "Prior"]
            trackers = [self.posterior_metric_tracker, self.metric_tracker]
        
        for name, tracker in zip(names, trackers):
            tracker.aggregate()
            _ = tracker.summary()
            fname = fname if fname is not None else self.results_name
            tracker.save_results(
                    exp_path=self.exp_path,
                    fname=f"{name}_{fname}"
                )
            tracker.make_plots(
                start_idx=self.exp_params["prediction_params"]["num_context"],
                savepath=os.path.join(self.exp_path, "results", f"{name}_{fname}")
            )
        return



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
    parser.add_argument(
            "--post_only",
            help="""If provided, we only evaluate using the latent actions inferred 
                    by InvDyn. Otherwise, we also evaluate using several randomly
                    sampled latent actions""",
            default=False, action='store_true'
        )
    parser.add_argument(
            "--num_samples",
            help="Number of random samples to use for evauation",
            type=int, default=10
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
            "Starting evaluation of predictor and inverse dynamics modules",
            message_type="new_exp"
        )
    logger.log_arguments(args)

    print_("Initializing Evaluator...")
    evaluator = StochEvaluator(
            exp_path=exp_path,
            name_pred_exp=args.name_pred_exp,
            savi_ckpt=args.savi_ckpt,
            pred_ckpt=args.pred_ckpt,
            num_seed=args.num_seed,
            num_preds=args.num_preds,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            results_name=args.results_name,
            set_expert_policy=args.set_expert_policy,
            post_only=args.post_only
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



#
