""" 
Training an Oracle model that predicts the GT actions given
the object representations
"""

import argparse
import os
import torch

from base.basePredictorTrainer import BasePredictorTrainer
from base.baseEvaluator import BaseEvaluator
from data.load_data import unwrap_batch_data
from lib.callbacks import Callbacks
from lib.config import Config
from lib.logger import Logger, print_
from lib.loss import LossTracker
import lib.setup_model as setup_model
import lib.utils as utils



class Trainer(BasePredictorTrainer, BaseEvaluator):
    """ 
    Training an Oracle model that predicts the GT actions given
    the object representations
    """
    
    def __init__(self, exp_path, savi_ckpt, name_oracle_exp, num_expert_demos=-1):
        """ Simple dataset and model checks """
        self.savi_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_oracle_exp)
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError(f"Downstream {self.exp_path = } does not exist...")
        
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.num_expert_demos = num_expert_demos
        self.checkpoint = None
        self.resume_training = False
        
        # for compatibility
        self.checkpoint = None  
        self.resume_training = False
        self.exp_params["prediction_params"] = {
                "num_context": 1,
                "num_preds": 16,
            }
        self.predictor = torch.nn.Identity()
        
        # using expert policy and enforcing the specifie number of expert demos
        self.num_expert_demos = num_expert_demos
        print_(f" --> Using only {self.num_expert_demos} expert demonstrations...")
        self.exp_params["dataset"]["num_expert_demos"] = self.num_expert_demos                  

        # relevant paths
        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                
        return


    def setup_oracle(self):
        """
        Instanciating Oracle model
        """
        # instanciating policy model
        oracle = setup_model.setup_behavior_model(self.exp_params, key="behavior")
        utils.log_architecture(
                oracle,
                exp_path=self.exp_path,
                fname="architecture_oracle.txt"
            )
        optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
                exp_params=self.exp_params,
                model=oracle
            )
        self.oracle = oracle.eval().to(self.device)
        self.predictor = self.oracle  # to re-use the base save_ckpt
        self.epoch = 0
        self.optimizer = optimizer
        self.lr_warmup = lr_warmup
        self.scheduler = scheduler
        self.loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        self.callback_manager = Callbacks(trainer=self)
        self.callback_manager.initialize_callbacks(trainer=self)
        return

            
    def forward_loss_metric(self, batch_data, training=False, inference_only=False):
        """
        Computing a forwad pass through the model, loss computation and optimizations

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, actions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed
        """
        # fetching and checking data
        imgs, _, init_kwargs, others = unwrap_batch_data(self.exp_params, batch_data)
        imgs = imgs.to(self.device)
        target_actions = others.get("actions").to(self.device)
        num_frames = target_actions.shape[1]

        out_model = self.savi(imgs, num_imgs=num_frames, decode=False, **init_kwargs)
        slot_history = out_model["slot_history"]  # (B, num_frames, num_slots, slot_dim)
        pred_actions = self.oracle(slot_history)

        # Generating only model outputs
        if inference_only:
            return pred_actions, None
        
        # loss and optimization
        self.loss_tracker(
                pred_action_embs=pred_actions.clamp(-1, 1),
                target_action_embs=target_actions.clamp(-1, 1)
            ) 
        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.oracle.parameters(),
                        self.exp_params["training"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return pred_actions, loss


    @torch.no_grad()
    def visualizations(self, *_, **__):
        """ No visualizations for this training"""
        return    
    
        
        
if __name__ == "__main__":
    utils.clear_cmd()
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the SAVi exp directory where the Oracle exp is located",
            required=True
        )
    parser.add_argument(
            "--savi_ckpt",
            help="Name of the pretrained SAVi checkpoint to load",
            required=True
        )
    parser.add_argument(
            "--name_oracle_exp",
            help="Name of the Oracle experiment",
            required=True
        )
    parser.add_argument(
            "--num_expert_demos",
            help="Number of expert demonstrations to use for training. -1 means 'all'",
            default=-1, type=int
        )
    args = parser.parse_args()
 
    # sanity checks and processing arguments
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    args.savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    args.name_oracle_exp = "oracle/" + args.name_oracle_exp
    if not os.path.exists(os.path.join(exp_path, args.name_oracle_exp)):
        raise FileNotFoundError(F"Oracle Exp-dir {args.name_oracle_exp} does not exist...")
    
    # Oracle training
    logger = Logger(exp_path=f"{args.exp_directory}/{args.name_oracle_exp}")
    logger.log_info("Training Oracle Model", message_type="new_exp")
    print_("Initializing Oracle Trainer...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    trainer = Trainer(
            exp_path=args.exp_directory,
            savi_ckpt=args.savi_ckpt,
            name_oracle_exp=args.name_oracle_exp,
            num_expert_demos=args.num_expert_demos
        )
    print_("Setting up model, Action-Decoder, and optimizer")
    trainer.load_savi()
    trainer.setup_oracle()
    print_("Loading dataset...")
    trainer.load_data()    
    print_("Starting to train")
    trainer.training_loop()