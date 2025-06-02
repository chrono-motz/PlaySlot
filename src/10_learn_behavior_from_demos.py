"""
Jointly training the behavior cloning and action decoding models
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
import lib.utils as utils
import lib.visualizations as visualizations
from lib.setup_model import \
        setup_behavior_model, \
        save_checkpoint, \
        setup_optimizer



class Trainer(BasePredictorTrainer, BaseEvaluator):
    """
    Jointly training the behavior cloning and action decoding models
    """
    
    def __init__(self, exp_path, savi_ckpt, name_pred_exp, pred_ckpt,
                 name_beh_exp, num_expert_demos=-1):
        """ Module initializer """
        self.savi_exp_path = exp_path
        self.name_pred_exp = name_pred_exp
        self.name_beh_exp = name_beh_exp
        self.pred_exp_path = os.path.join(exp_path, name_pred_exp)
        self.exp_path = os.path.join(exp_path, name_pred_exp, name_beh_exp)
        if not os.path.exists(self.savi_exp_path):
            raise FileNotFoundError(f"SAVi-Exp {self.savi_exp_path} does not exist.")
        if not os.path.exists(self.pred_exp_path):
            raise FileNotFoundError(f"Pred-Exp {self.pred_exp_path} does not exist.")
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError(f"Beh-Exp {self.exp_path} does not exist.")        
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.pred_ckpt = pred_ckpt
        self.set_expert_policy = True  # always trained with expert demos
        
        # for compatibility
        self.checkpoint = None  
        self.resume_training = False
        self.exp_params["prediction_params"] = {
                "num_context": 1,
                "num_preds": 16,
            }
      
        # using expert policy and enforcing the specifie number of expert demos
        self.num_expert_demos = num_expert_demos
        print_(f" --> Using only {self.num_expert_demos} expert demonstrations...")
        self.exp_params["dataset"]["num_expert_demos"] = self.num_expert_demos            

        # creating path for models, tboard and so on
        self.savi_models_path = os.path.join(self.savi_exp_path, "models")
        self.pred_models_path = os.path.join(self.pred_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        tboard = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard)
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return
    
    
    def setup_behavior_models(self):
        """
        Instanciating dowsntream models, i.e. Behavior-Cloning and Action Decoding
        """
        # instanciating action decoder
        action_decoder = setup_behavior_model(self.exp_params, key="action")
        utils.log_architecture(
                action_decoder,
                exp_path=self.exp_path,
                fname="architecture_action_decoder.txt"
            )
        optimizer, _, _ = setup_optimizer(
                exp_params=self.exp_params,
                model=action_decoder
            )
        self.action_decoder = action_decoder.eval().to(self.device)
        self.action_optimizer = optimizer
        
        # instanciating policy model
        policy_model = setup_behavior_model(self.exp_params, key="behavior")
        utils.log_architecture(
                policy_model,
                exp_path=self.exp_path,
                fname="architecture_policy_model.txt"
            )
        optimizer, scheduler, lr_warmup = setup_optimizer(
                exp_params=self.exp_params,
                model=policy_model
            )
        self.policy_model = policy_model.eval().to(self.device)
        self.policy_optimzier = optimizer

        self.epoch = 0
        self.optimizer = optimizer
        self.lr_warmup = lr_warmup
        self.scheduler = scheduler
        self.loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        self.callback_manager = Callbacks(trainer=self)
        self.callback_manager.initialize_callbacks(trainer=self)
        return


    def wrapper_save_checkpoint(self, epoch=None, savedir="models",
                                savename=None, finished=False):
        """
        Overriding the saving wrapper for saving both the policy model
        and the action decoder
        """
        save_checkpoint(
                model=self.action_decoder,
                optimizer=self.action_optimizer,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir=savedir,
                savename=savename,
                finished=finished,
                prefix="ActDec_"
            )
        save_checkpoint(
                model=self.policy_model,
                optimizer=self.policy_optimzier,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir=savedir,
                savename=savename,
                finished=finished,
                prefix="Policy_"
            )
        return


    def forward_loss_metric(self, batch_data, training=False, inference_only=False,):
        """
        Computing a forwad pass through the model, loss values and optimization

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
        imgs, _, init_kwargs, metas = unwrap_batch_data(self.exp_params, batch_data)
        num_frames = imgs.shape[1]
        imgs = imgs.to(self.device)
        inverse_dynamics_model = self.predictor.latent_action

        # encoding frames into object slots and computing Latent Action vectors
        with torch.no_grad():
            out_model = self.savi(
                    imgs,
                    num_imgs=num_frames,
                    decode=False,
                    **init_kwargs
                )
            slot_history = out_model["slot_history"]
            out_dict = inverse_dynamics_model.compute_actions(slot_history)
            target_latent_actions = out_dict.get("sampled_latent_action")

        # Predicting latent actions with policy and decoding to real action space
        # We remove the last time-step, which was only needed for inverse dynamics
        causal_slots = slot_history[:, :-1].detach()
        pred_latent_actions = self.policy_model(causal_slots)
        # pred_actions = self.action_decoder(pred_latent_actions.detach())
        pred_actions = self.action_decoder(pred_latent_actions.detach())
        out = (pred_latent_actions, pred_actions)

        # Generating only model outputs
        if inference_only:
            return out, None
        
        # target actions
        target_actions = metas["actions"].to(self.device)
        if self.exp_params["dataset"]["dataset_name"] in [
                    "BlockPush",
                    "BlockPush_ExpertDemos"
                ]:  # BlockPush has one action too many
            target_actions = target_actions[:, :-1]

        # if necessary: loss computation, backward pass and optimization
        self.loss_tracker(
                pred_action_embs=pred_latent_actions,
                target_action_embs=target_latent_actions.detach(),
                pred_actions=pred_actions,
                target_actions=target_actions
            )
        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.action_optimizer.zero_grad()
            self.policy_optimzier.zero_grad()
            loss.backward()
            self.action_optimizer.step()
            self.policy_optimzier.step()

        return out, loss


    @torch.no_grad()
    def inference(self, imgs, max_steps=30):
        """
        Running inference through the behavior cloning model in order to
        solve the task autoregressively in the model latent imagination.

        Args:
        -----
        imgs: torch tensor
            Images from the environment. We will only use the first frame as seed.
            Shape is (B, num_frames, 3, H, W)
        max_steps: int
            Number of steps to predict for in order to solve the task.
        """
        InvDyn = self.predictor.latent_action
        cOCVP = self.predictor.dynamics_model
        
        # computing initial object slots
        out_model = self.savi(imgs[:, :1], num_imgs=1, decode=False)
        init_slots = out_model["slot_history"]
        num_slots = init_slots.shape[-2]

        # autoregressively predicting actions and slots for 'max_steps'
        # within the model's latent imagination
        cur_slots = init_slots
        all_pred_slots = [cur_slots]
        all_latent_actions, all_action_protos, all_action_vars = [], [], []
        for i in range(max_steps):
            # predicting latent action with policy 
            latent_action = self.policy_model(cur_slots[:, i:i+1])[:, 0]
            # parsing latent action into prootype and variability
            action_proto, action_var = InvDyn.decompose_action_latent(latent_action)
            all_latent_actions.append(latent_action)
            all_action_protos.append(action_proto)
            all_action_vars.append(action_var)

            # predicting next slots conditioned on action
            cur_actions_protos = torch.stack(all_action_protos, dim=1)
            cur_action_vars = torch.stack(all_action_vars, dim=1)
            cur_slots, cur_actions_protos, cur_action_vars = cOCVP.enforce_window(
                    slots=cur_slots,
                    action_protos=cur_actions_protos,
                    action_vars=cur_action_vars
                )
            if len(cur_actions_protos.shape) == 3:  # inflating to all slots
                cur_actions_protos = cur_actions_protos.unsqueeze(2)
                cur_actions_protos = cur_actions_protos.expand(-1, -1, num_slots, -1)
                cur_action_vars = cur_action_vars.unsqueeze(2)
                cur_action_vars = cur_action_vars.expand(-1, -1, num_slots, -1)
            pred_slots = cOCVP.forward_single(
                    slots=cur_slots,
                    action_protos=cur_actions_protos,
                    action_vars=cur_action_vars
                )[:, -1:]
            all_pred_slots.append(pred_slots)
            cur_slots = torch.cat(all_pred_slots, dim=1)

        # decoding predicted slots
        all_pred_slots = torch.cat(all_pred_slots, dim=1)
        B, num_frames, num_slots, slot_dim = all_pred_slots.shape
        all_pred_slots = all_pred_slots.reshape(B * num_frames, num_slots, slot_dim)
        recons_imgs, (pred_recons, pred_masks) = self.savi.decode(all_pred_slots)
    
        recons_imgs = recons_imgs.reshape(B, num_frames, *recons_imgs.shape[1:])
        pred_recons = pred_recons.reshape(B, num_frames, *pred_recons.shape[1:])
        pred_masks = pred_masks.reshape(B, num_frames, *pred_masks.shape[1:])
        inference_out = {
            "pred_imgs": recons_imgs,
            "pred_objs": pred_recons,
            "pred_masks": pred_masks
        }
        return inference_out
        

    @torch.no_grad()
    def visualizations(self, batch_data, epoch, iter_):
        """
        Making a visualization of TBD
        """
        if(iter_ % self.exp_params["training"]["image_log_frequency"] != 0):
            return
        imgs, _, _, _ = unwrap_batch_data(self.exp_params, batch_data)
        imgs = imgs.to(self.device)
        inference_out = self.inference(imgs, max_steps=24)
        pred_imgs = inference_out["pred_imgs"]
        pred_masks = inference_out["pred_masks"]
        pred_objs = inference_out["pred_objs"]
        
        N = min(3, imgs.shape[0])
        ids = torch.arange(0, N)  # first videos in batch
        for idx in range(N):
            k = ids[idx]
            
            _ = visualizations.visualize_recons(
                imgs=imgs[k][1:],
                recons=pred_imgs[k].clamp(0, 1),
                tag=f"_imgs_{idx}",
                savepath=None,
                tb_writer=self.writer,
                iter=iter_
            )

            # Rendered individual object masks
            _ = visualizations.visualize_decomp(
                    pred_masks[k][:10].clamp(0, 1),
                    savepath=None,
                    tag=f"slot_masks_{idx}",
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    tb_writer=self.writer,
                    iter=iter_,
                    n_cols=min(10, pred_masks.shape[1])
                )

            # Rendered individual combination of an object with its masks
            recon_combined = pred_objs[k][:10] * pred_masks[k][:10]
            _ = visualizations.visualize_decomp(
                    recon_combined.clamp(0, 1),
                    savepath=None,
                    tag=f"slot_combined_{idx}",
                    vmin=0,
                    vmax=1,
                    tb_writer=self.writer,
                    iter=iter_,
                    n_cols=min(10, pred_masks.shape[1])
                )



if __name__ == "__main__":
    utils.clear_cmd()
    
    # processing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the father SAVi exp. directory",
            required=True
        )
    parser.add_argument(
            "--savi_ckpt",
            help="Name of the pretrained SAVi checkpoint to use",
            required=True
        )
    parser.add_argument(
            "--name_pred_exp",
            help="Name of the predictor exp_directory.",
            required=True
        )
    parser.add_argument(
            "--pred_ckpt",
            help="Name of the pretrained PlaySlot predictor checkpoint to load",
            required=True
        )
    parser.add_argument(
            "--name_beh_exp",
            help="Name of the behavior experiment to train.",
            required=True
        )
    parser.add_argument(
            "--num_expert_demos",
            help="Number of expert demos to use for training. -1 mean 'use all'",
            default=-1,
            type=int
        )
    args = parser.parse_args()
    
    # SAVi exp checks    
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    args.savi_model = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    
    # predictor and behavior exp checks
    args.name_pred_exp = utils.process_predictor_experiment(
            exp_directory=exp_path,
            name_predictor_experiment=args.name_pred_exp,    
        )
    pred_exp_path = os.path.join(exp_path, args.name_pred_exp)
    args.pred_ckpt = utils.process_checkpoint_argument(
            exp_path=pred_exp_path,
            checkpoint=args.pred_ckpt
        )
    args.name_beh_exp = utils.process_behavior_experiment(
            exp_directory=pred_exp_path,
            name_behavior_experiment=args.name_beh_exp,
        )
    
    # Starting Behavior Training
    logger = Logger(
            exp_path=f"{args.exp_directory}/{args.name_pred_exp}/{args.name_beh_exp}"
        )
    logger.log_info(
            "Starting Training of Policy Model and Action Decoder",
            message_type="new_exp"
        )
    print_("Initializing Trainer...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    trainer = Trainer(
            exp_path=args.exp_directory,
            savi_ckpt=args.savi_ckpt,
            name_pred_exp=args.name_pred_exp,
            pred_ckpt=args.pred_ckpt,
            name_beh_exp=args.name_beh_exp,
            num_expert_demos=args.num_expert_demos,
        )

    print_("Loading dataset...")
    trainer.load_data()
    print_("Loading pretrained SAVi and PlaySlot models")
    trainer.load_savi(models_path=trainer.savi_models_path)
    trainer.load_predictor(models_path=trainer.pred_models_path)
    print_("Setting up Behavior Models...")
    trainer.setup_behavior_models()
    print_("Starting to train")
    trainer.training_loop()


#
