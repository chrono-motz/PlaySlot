"""
Training and Validation of an Object-Centric Video Prediction module given a
pretrained SAVI video decomposition model.

This script can be used to train the following OCVP-Models:
  - Vanilla OCVP (same as SlotFormer)
  - OCVP-Seq
  - OCVP-Par
  - Action-Conditional OCVP
"""

import argparse
import torch

from base.basePredictorTrainer import BasePredictorTrainer
from data.load_data import unwrap_batch_data
from lib.logger import Logger, print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_qualitative_eval



class Trainer(BasePredictorTrainer):
    """
    Training and Validation of an Object-Centric Video Prediction module given a
    pretrained SAVI video decomposition model.
    """

    def forward_loss_metric(self, batch_data, training=False,
                            inference_only=False, **kwargs):
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

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]

        # fetching and checking data
        videos, _, init_kwargs, others = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq_len {L} smaller that {num_context + num_preds = }")

        # encoding frames into object slots usign pretrained SAVi
        with torch.no_grad():
            out_model = self.savi(videos, num_imgs=L, decode=False, **init_kwargs)
            slot_history = out_model["slot_history"]

        # predicting future slots
        actions = others.get("actions", None)
        pred_slots, _ = self.predictor(
            slot_history=slot_history,
            use_posterior=self.predictor.training,
            actions=actions,
            num_seed=num_context,
            num_preds=num_preds
        )
        # rendering future objects and frames from predicted object slots
        num_frames = num_context + num_preds - 1
        num_slots, slot_dim = pred_slots.shape[-2], pred_slots.shape[-1]
        pred_slots_decode = pred_slots.clone().reshape(B * num_frames, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.savi.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_frames, C, H, W)

        # Generating only model outputs
        out_model = (pred_imgs, pred_recons, pred_masks)
        if inference_only:
            return out_model, None

        # loss computation, backward pass and optimization
        target_slots = slot_history[:, 1:num_context+num_preds, :, :]
        target_imgs = videos[:, 1:num_context+num_preds, :, :]
        self.loss_tracker(
                preds=pred_slots,
                targets=target_slots,
                pred_imgs=pred_imgs,
                target_imgs=target_imgs,
            )
        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimization(loss=loss, iter_=self.iter_, epoch=self.epoch)
        return out_model, loss


    @torch.no_grad()
    def visualizations(self, batch_data, epoch, iter_):
        """
        Making a visualization of some ground-truth, targets and predictions
        from the current model.
        """
        if(iter_ % self.exp_params["training"]["image_log_frequency"] != 0):
            return

        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]

        # forward pass
        videos, _, _, _ = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        pred_imgs, pred_recons, pred_masks = out_model
        pred_imgs = pred_imgs[:, num_context-1:num_context+num_preds-1]
        pred_recons = pred_recons[:, num_context-1:num_context+num_preds-1]
        pred_masks = pred_masks[:, num_context-1:num_context+num_preds-1]
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]

        # visualitations
        N = min(3, videos.shape[0])
        ids = torch.arange(0, N)  # first videos in batch
        for idx in range(N):
            k = ids[idx]
            fig, _ = visualize_qualitative_eval(
                context=videos[k, :num_context],
                targets=target_imgs[k],
                preds=pred_imgs[k],
                savepath=None
            )
            self.writer.add_figure(tag=f"Qualitative Eval {k+1}", figure=fig, step=iter_)

            objs = pred_masks[k*num_preds:(k+1)*num_preds] * pred_recons[k*num_preds:(k+1)*num_preds]
            _ = visualize_decomp(
                    objs.clamp(0, 1),
                    savepath=None,
                    tag=f"Pred. Object Recons. {k+1}",
                    tb_writer=self.writer,
                    iter=iter_
                )
            _ = visualize_decomp(
                    pred_masks[k*num_preds:(k+1)*num_preds],
                    savepath=None,
                    tag=f"Pred. Masks. {k+1}",
                    iter=iter_,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    tb_writer=self.writer
                )
        return



if __name__ == "__main__":
    utils.clear_cmd()
    
   # process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the SAVi exp. directory. It includes the predictor exp.",
            required=True
        )
    parser.add_argument(
            "--savi_ckpt",
            help="Name of the pretrained SAVi checkpoint to use",
            required=True
        )
    parser.add_argument(
            "--name_pred_exp",
            help="Name to the predictor experiment to train.",
            required=True
        )
    # for resuming training
    parser.add_argument(
            "--checkpoint",
            help="Checkpoint with predictor pretrained parameters to load",
            default=None
        )
    parser.add_argument(
            "--resume_training",
            help="Resuming training",
            default=False,
            action='store_true'
        )
    args = parser.parse_args()
    
    # sanity checks on arguments
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    args.name_pred_exp = utils.process_predictor_experiment(
            exp_directory=exp_path,
            name_predictor_experiment=args.name_pred_exp,    
        )
    args.checkpoint = utils.process_predictor_checkpoint(
            exp_path=exp_path,
            name_predictor_experiment=args.name_pred_exp,
            checkpoint=args.checkpoint
        )    
    
    # Training
    logger = Logger(exp_path=f"{exp_path}/{args.name_pred_exp}")
    logger.log_info("Starting OCVP training", message_type="new_exp")
    logger.log_arguments(args)

    print_("Initializing Trainer...")
    trainer = Trainer(
            name_pred_exp=args.name_pred_exp,
            exp_path=exp_path,
            savi_ckpt=args.savi_ckpt,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Loading dataset...")
    trainer.load_data()
    print_("Setting up model, predictor and optimizer")
    trainer.load_savi()
    trainer.setup_predictor()
    print_("Starting to train")
    trainer.training_loop()


