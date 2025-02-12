"""
Base evaluator from which all backbone evaluator modules inherit.

Basically it removes the scaffolding that is repeat across all evaluation modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import log_function, for_all_methods, print_
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class BaseEvaluator:
    """
    Base Class for evaluating a model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    checkpoint: string/None
        Name of a model checkpoint to evaluate.
        It must be stored in the models/ directory of the experiment directory.
    """

    def __init__(self, exp_path, savi_ckpt, batch_size=None,
                 results_name=None, set_expert_policy=False):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.batch_size = batch_size
        model_name = savi_ckpt.split(".")[0]
        self.results_name = f"{model_name}" if results_name is None else results_name
        self.set_expert_policy = set_expert_policy

        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        return

    def set_expert_policy_dataset(self):
        """ 
        Exchanging the current dataset with a variant using an ExpertPolicy
        """
        # updating results file name
        if hasattr(self, "results_name") and self.results_name is not None:
            self.results_name = self.results_name + "_ExpertPolicy"
        
        print_(f"Using Expert Policy dataset variant:")
        data.set_expert_policy_dataset(db_params=self.exp_params["dataset"])
        return


    def set_metric_tracker(self):
        """ Initializing the metric tracker """
        metric_tracker = MetricTracker(
                self.exp_path,
                metrics=["psnr", "ssim", "lpips"]
            )
        return metric_tracker


    def override_num_seed_and_preds(self, num_seed=None, num_preds=None):
        """
        Overrriding the 'num_seed' and 'num_preds' parameters in the experiment
        parameters with the values provided as arguments.
        """
        # overriding 'num_seed' if given as argument
        if num_seed is not None:
            npreds_tmp = self.exp_params["prediction_params"]["num_preds"]
            self.exp_params["prediction_params"]["num_context"] = num_seed
            self.exp_params["prediction_params"]["sample_length"] = num_seed + npreds_tmp + 1
            print_(f"  --> Overriding 'num_context' to {num_seed}")
            print_(f"  --> New 'sample_length' is {num_seed + npreds_tmp}")

        # overriding 'num_preds' if given as argument
        if num_preds is not None:
            nseed_tmp = self.exp_params["prediction_params"]["num_context"]
            self.exp_params["prediction_params"]["num_preds"] = num_preds
            self.exp_params["prediction_params"]["sample_length"] = nseed_tmp + num_preds + 1
            print_(f"  --> Overriding 'num_preds' to {num_preds}")
            print_(f"  --> New 'sample_length' is {nseed_tmp + num_preds}")

        # updating sequence length from dataset given updated paramters
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        new_seq_len = num_context + num_preds
        print_(f"Replacing sequence length with required seq. length of {new_seq_len}")
        self.exp_params["dataset"]["num_frames"] = new_seq_len
        return


    def load_data(self):
        """
        Updating data parameters and loading test-set and data loader
        """
        # updating batch size
        self.batch_size = getattr(self, "batch_size", None)
        if self.batch_size is not None and self.batch_size > 0:
            cur_batch_size = self.exp_params["training"]["batch_size"]
            print_(f"Overriding BATCH-SIZE from {cur_batch_size} to {self.batch_size}")
            self.exp_params["training"]["batch_size"] = self.batch_size
        
        # setting up expert policy        
        if self.set_expert_policy:
            self.set_expert_policy_dataset()
        
        # instanciating test dataset and data-loader
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        self.test_set = data.load_data(exp_params=self.exp_params, split="test")
        self.test_loader = data.build_data_loader(
                dataset=self.test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return


    def load_savi(self, models_path=None):
        """
        Initializing model and loading pretrained parameters given checkpoint
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        savi = setup_model.setup_model(model_params=self.exp_params["model"])
        savi = savi.eval().to(self.device)

        # loading pretrained paramters
        models_path = self.models_path if models_path is None else models_path
        checkpoint_path = os.path.join(models_path, self.savi_ckpt)
        savi = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=savi,
                only_model=True
            )
        self.savi = savi.eval()
        self.metric_tracker = self.set_metric_tracker()
        return


    def load_predictor(self, models_path=None):
        """
        Load pretrained predictor model from the corresponding model checkpoint
        """
        predictor = setup_model.setup_predictor(exp_params=self.exp_params)
        predictor = predictor.eval().to(self.device)

        print_(f"Loading pretrained parameters from checkpoint {self.pred_ckpt}...")
        models_path = self.models_path if models_path is None else models_path
        predictor = setup_model.load_checkpoint(
                checkpoint_path=os.path.join(models_path, self.pred_ckpt),
                model=predictor,
                only_model=True,
            )
        self.predictor = predictor.eval()
        return


    @torch.no_grad()
    def evaluate(self):
        """
        Evaluating model
        """
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for i, batch_data in progress_bar:
            self.forward_eval(batch_data=batch_data)
            progress_bar.set_description(f"Iter {i}/{len(self.test_loader)}")
        self.aggregate_and_save_metrics()
        return
    

    @torch.no_grad()
    def generate_figs(self):
        """
        Computing and saving visualizations
        """
        progress_bar = tqdm(enumerate(self.test_loader), total=self.num_seqs)
        for i, batch_data in progress_bar:
            if i >= self.num_seqs:
                break
            self.compute_visualization(batch_data=batch_data, img_idx=i)
        return


    @torch.no_grad()
    def aggregate_and_save_metrics(self, fname=None):
        """
        Aggregating all computed metrics and saving results to logs file
        """
        self.metric_tracker.aggregate()
        self.results = self.metric_tracker.summary()
        fname = fname if fname is not None and isinstance(fname, str) else self.results_name
        self.metric_tracker.save_results(exp_path=self.exp_path, fname=fname)
        
        # plots with per-frame metrics, but only for predictors
        if "prediction_params" in self.exp_params:
            self.metric_tracker.make_plots(
                start_idx=self.exp_params["prediction_params"]["num_context"],
                savepath=os.path.join(self.exp_path, "results", fname)
            )
        return


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
        raise NotImplementedError("Base Evaluator Module does not implement 'forward_eval'...")


    def compute_visualization(self, batch_data, img_idx, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        img_idx: int
            Index of the visualization to compute and save
        """
        raise NotImplementedError("Base FigGenerator does not implement 'compute_visualization'...")


#
