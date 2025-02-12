"""
Training and Validating a SAVi video decomposition model
"""

import argparse
import torch

from base.baseTrainer import BaseTrainer
from data.load_data import unwrap_batch_data
from lib.logger import Logger, print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_recons



class Trainer(BaseTrainer):
    """
    Class for training a SAVi model for object-centric video
    """

    def forward_loss_metric(self, batch_data, training=False, inference_only=False):
        """
        Computing a forwad pass through the model,
        and (if necessary) the loss values and metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, actions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        videos, _, initializer_kwargs, _ = unwrap_batch_data(self.exp_params, batch_data)

        # forward pass
        videos = videos.to(self.device)
        out_model = self.model(videos, num_imgs=videos.shape[1], **initializer_kwargs)
        recons_imgs = out_model.get("recons_imgs")

        if inference_only:
            return out_model, None

        # loss computation, backward pass and optimization
        self.loss_tracker(
                pred_imgs=recons_imgs,
                target_imgs=videos.clamp(0, 1),
            )

        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss

    @torch.no_grad()
    def visualizations(self, batch_data, iter_):
        """
        Making a visualization of some ground-truth, targets and predictions
        from the current model, and logging them to tensorboard
        """
        if(iter_ % self.exp_params["training"]["image_log_frequency"] != 0):
            return

        videos, _, _, _ = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        N = min(10, videos.shape[1])  # max of 10 frames for sleeker figures

        recons_history = out_model.get("recons_imgs")
        recons_objs = out_model.get("recons_objs")
        recons_masks = out_model.get("masks")

        # output reconstructions and input images
        visualize_recons(
                imgs=videos[0][:N],
                recons=recons_history[0][:N].clamp(0, 1),
                tag="_recons", 
                savepath=None,
                tb_writer=self.writer,
                iter=iter_
            )

        # Rendered individual object masks
        _ = visualize_decomp(
                recons_masks[0][:N].clamp(0, 1),
                savepath=None,
                tag="slot_masks",
                cmap="gray",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_,
                n_cols=N
            )

        # Rendered individual combination of an object with its masks
        recon_combined = recons_objs[0][:N] * recons_masks[0][:N]
        _ = visualize_decomp(
                recon_combined.clamp(0, 1),
                savepath=None,
                tag="slot_combined",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_,
                n_cols=N
            )
        return



if __name__ == "__main__":
    utils.clear_cmd()
    
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the experiment directory",
            required=True
        )
    parser.add_argument(
            "--checkpoint",
            help="Checkpoint with pretrained parameters to load",
            default=None
        )
    parser.add_argument(
            "--resume_training",
            help="For resuming training",
            default=False,
            action='store_true'
        )
    args = parser.parse_args()
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    checkpoint = utils.process_checkpoint_argument(exp_path, args.checkpoint)
    resume_training = args.resume_training
    
    # training
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting SAVi training procedure", message_type="new_exp")
    logger.log_arguments(args)

    print_("Initializing SAVi Trainer...")
    trainer = Trainer(
            exp_path=exp_path,
            checkpoint=checkpoint,
            resume_training=args.resume_training
        )
    print_("Setting up model and optimizer")
    trainer.setup_model()
    print_("Loading dataset...")
    trainer.load_data()
    print_("Starting to train")
    trainer.training_loop()

