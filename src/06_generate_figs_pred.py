"""
Generating some video prediction, object prediction and segmentation figures
using a pretrained SAVi model and the corresponding predictor, which can be
an OCVP, PlaySlot, or ActionCondOCVP model.
"""

import argparse
import os
import torch
from tqdm import tqdm

from base.baseEvaluator import BaseEvaluator
from data.load_data import unwrap_batch_data
from lib.config import Config
from lib.logger import print_
from lib.metrics import MetricTracker
import lib.utils as utils
import lib.visualizations as vis



class FigGenerator(BaseEvaluator):
    """
    Generating some video prediction, object prediction and segmentation figures
    using a pretrained SAVi model and the corresponding predictor, which can be
    an OCVP, PlaySlot, or ActionCondOCVP model.
    """

    def __init__(self, exp_path, name_pred_exp, savi_ckpt, pred_ckpt, 
                 num_seqs=10, num_seed=None, num_preds=None,
                 set_expert_policy=False):
        """
        Initializing the trainer object
        """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.pred_ckpt = pred_ckpt
        self.num_seqs = num_seqs
        self.set_expert_policy = set_expert_policy
        self.override_num_seed_and_preds(num_seed=num_seed, num_preds=num_preds)
        self.savi_model_path = os.path.join(self.parent_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")

        # creating directory where to store the figures
        self.pred_name = self.exp_params["predictor"]["predictor_name"]
        self.plots_dir = \
                f"FigGen_Pred_{self.pred_name}_{name_pred_exp.split('/')[-1]}_" + \
                f"{pred_ckpt[:-4]}_" + \
                f"NumSeed={num_seed}_NumPreds={num_preds}_" + \
                f"_ExpertPolicy={set_expert_policy}"
        self.plots_path = os.path.join(self.exp_path, "plots", self.plots_dir)
        utils.create_directory(self.plots_path)
        return
    

    @torch.no_grad()
    def generate_figs(self):
        """
        Generating figures
        """
        utils.set_random_seed()
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        metric_tracker = MetricTracker(exp_path=None, metrics=["psnr", "lpips"])

        for idx in tqdm(range(self.num_seqs)):
            # loading sequence and sanity checks
            batch_data = self.test_set[idx]
            videos, _, init_data, other = unwrap_batch_data(self.exp_params, batch_data)
            videos = videos.unsqueeze(0).to(self.device)
            init_data = {
                    k: v.unsqueeze(0) for k, v in init_data.items() if torch.is_tensor(v)
                }
            other = {
                    k: v.unsqueeze(0) for k, v in other.items() if torch.is_tensor(v)
                }
            n_frames = videos.shape[1]
            if n_frames < num_context + num_preds:
                raise ValueError(f"{n_frames = } < {num_context + num_preds = }")

            # forward pass through SAVi and predictor models
            out_model = self.forward_pass(
                    videos=videos,
                    init_data=init_data,
                    other=other
                )
            pred_imgs, pred_objs, pred_masks, recons_objs, recons_masks = out_model

            # computing metrics for sequence to visualize and creating out_dor
            metric_tracker.reset_results()
            metric_tracker.accumulate(
                    preds=pred_imgs.clamp(0, 1),
                    targets=videos[:1, num_context:num_context+num_preds].clamp(0, 1)
                )
            metric_tracker.aggregate()
            results = metric_tracker.get_results()
            psnr, lpips = results["psnr"]["mean"], results["lpips"]["mean"]
            cur_dir = f"img_{idx+1}_psnr={round(psnr,2)}_lpips={round(lpips, 3)}"

            # generating and saving visualizations
            self.compute_visualization(
                    videos=videos,
                    recons_objs=recons_objs,
                    recons_masks=recons_masks,
                    pred_imgs=pred_imgs,
                    pred_objs=pred_objs,
                    pred_masks=pred_masks,
                    out_dir=cur_dir
                )
        return


    @torch.no_grad()
    def forward_pass(self, videos, init_data, other):
        """
        Forward pass through SAVi and Preditor
        """
        # computing object slots and decoding into object images
        B, L, C, H, W = videos.shape
        out_model = self.savi(videos, num_imgs=L, **init_data)
        slot_history = out_model.get("slot_history")
        recons_objs = out_model.get("recons_objs")
        recons_masks = out_model.get("masks")

        # predicting future slots
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        actions = other.get("actions").to(self.device) if "actions" in other else None
        pred_slots, _ = self.predictor(
                slot_history,
                use_posterior=True,
                actions=actions,
                num_seed=num_context,
                num_preds=num_preds
            )
        if pred_slots.shape[1] > num_preds:  # keeping only pred frames
            pred_slots = pred_slots[:, -num_preds:]
        
        # decoding predicted slots into predicted frames and object images
        num_slots, slot_dim = pred_slots.shape[-2], pred_slots.shape[-1]
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_objs, pred_masks) = self.savi.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_preds, C, H, W)
        pred_objs = pred_objs.view(B, num_preds, num_slots, *pred_objs.shape[-3:])
        pred_masks = pred_masks.view(B, num_preds, num_slots, *pred_masks.shape[-3:])

        return pred_imgs, pred_objs, pred_masks, recons_objs, recons_masks


    def compute_visualization(self, videos, recons_objs, recons_masks, pred_imgs,
                              pred_objs, pred_masks, out_dir):
        """
        Saving predicted images, masks, objects and segmentations
        """
        utils.create_directory(self.plots_path, out_dir)

        # some hpyer-parameters of the video model
        B = videos.shape[0]
        num_slots = self.savi.num_slots
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        seed_imgs = videos[:, :num_context, :, :]
        seed_objs = recons_objs[:, :num_context]
        seed_masks = recons_masks[:, :num_context]
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]

        # aligned objects (seed and pred)
        seed_objs = vis.add_border(seed_objs * seed_masks, color_name="green", pad=2)
        pred_objs = vis.add_border(pred_objs * pred_masks, color_name="red", pad=2)
        all_objs = torch.cat([seed_objs, pred_objs], dim=1)
        _ = vis.visualize_aligned_slots(
                all_objs[0],
                savepath=os.path.join(self.plots_path, out_dir, "aligned_slots.png")
            )

        # Video predictions
        _ = vis.visualize_qualitative_eval(
                context=seed_imgs[0],
                targets=target_imgs[0],
                preds=pred_imgs[0],
                savepath=os.path.join(self.plots_path, out_dir, "qual_eval_rgb.png")
            )

        # masks as segmentations
        seed_masks_categorical = seed_masks[0].argmax(dim=1)
        pred_masks_categorical = pred_masks[0].argmax(dim=1)
        all_masks_categorical = torch.cat(
                [seed_masks_categorical, pred_masks_categorical],
                dim=0
            )
        masks_vis = vis.masks_to_rgb(x=all_masks_categorical)[:, 0]

        # overlaying masks on images
        masks_categorical_channels = vis.idx_to_one_hot(x=all_masks_categorical[:, 0])
        disp_overlay = vis.overlay_segmentations(
                videos[0].cpu().detach(),
                masks_categorical_channels.cpu().detach(),
                colors=vis.COLORS,
                alpha=0.6
            )

        # Sequence GIFs
        gt_frames = torch.cat([seed_imgs, target_imgs], dim=1)
        pred_frames = torch.cat([seed_imgs, pred_imgs], dim=1)
        vis.make_gif(
                gt_frames[0], n_seed=1000, use_border=True,
                savepath=os.path.join(self.plots_path, out_dir, "gt_GIF_frames.gif"),
            )
        vis.make_gif(
                pred_frames[0], n_seed=num_context, use_border=True,
                savepath=os.path.join(self.plots_path, out_dir, "pred_GIF_frames.gif"),
            )
        vis.make_gif(
                masks_vis, n_seed=num_context, use_border=True,
                savepath=os.path.join(self.plots_path, out_dir, "masks_GIF_masks.gif"),
            )
        vis.make_gif(
                disp_overlay, n_seed=num_context, use_border=True,
                savepath=os.path.join(self.plots_path, out_dir, "overlay_GIF.gif"),
            )

        # Object GIFs
        for obj_id in range(all_objs.shape[2]):
            vis.make_gif(
                all_objs[0, :, obj_id], n_seed=num_context, use_border=False,
                savepath=os.path.join(self.plots_path, out_dir, f"obj_{obj_id+1}.gif"),
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
    # additional arguments
    parser.add_argument(
            "--num_seqs",
            help="Number of sequences to generate",
            type=int, default=10
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
            "--set_expert_policy",
            help="If given, expert policy variant is used...",
            default=False, action='store_true'
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
    
    # Figure generation
    print_("Generating figures for predictor model...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            name_pred_exp=args.name_pred_exp,
            savi_ckpt=args.savi_ckpt,
            pred_ckpt=args.pred_ckpt,
            num_seqs=args.num_seqs,
            num_seed=args.num_seed,
            num_preds=args.num_preds,
            set_expert_policy=args.set_expert_policy,
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up SAVi and predictor and loading pretrained parameters")
    figGenerator.load_savi(models_path=figGenerator.savi_model_path)
    figGenerator.load_predictor(models_path=figGenerator.models_path)    
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
