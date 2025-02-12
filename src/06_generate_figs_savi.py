"""
Generating figures using a pretrained SAVI model
"""

import argparse
import os
import torch

from base.baseEvaluator import BaseEvaluator
from data.load_data import unwrap_batch_data
from lib.config import Config
from lib.logger import print_
import lib.utils as utils
from lib.visualizations import visualize_recons, visualize_decomp



class FigGenerator(BaseEvaluator):
    """
    Class for generating figures using a pretrained SAVI model
    """

    def __init__(self, exp_path, savi_ckpt, num_seqs=10, num_frames=None,
                 set_expert_policy=False):
        """
        Initializing the figure generation module
        """
        self.exp_path = os.path.join(exp_path)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt 
        self.num_seqs = num_seqs 
        self.num_frames = num_frames 
        self.set_expert_policy = set_expert_policy 

        # direcoty where the figures will be saved
        model_name = savi_ckpt.split('.')[0]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"figGeneration_SAViModel={model_name}" + 
                f"_NumFrames={num_frames}" + 
                f"_ExpertPolicy={set_expert_policy}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.plots_path)
        
        # overriding sequence length of dataset with argument
        if num_frames is not None:
            old_num_frames = self.exp_params["dataset"]["num_frames"]
            print_(f"Overriding number of frames from {old_num_frames} to {num_frames}")
            self.exp_params["dataset"]["num_frames"] = num_frames
        return


    @torch.no_grad()
    def compute_visualization(self, batch_data, img_idx):
        """
        Computing visualization
        """
        videos, _, init_kwargs, _ = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        out_model = self.savi(videos, num_imgs=videos.shape[1], **init_kwargs)
        recons_history = out_model.get("recons_imgs")
        recons_objs = out_model.get("recons_objs")
        recons_masks = out_model.get("masks")

        # directories for saving figures
        cur_dir = f"sequence_{img_idx:02d}"
        utils.create_directory(os.path.join(self.plots_path, cur_dir))

        # saving the reconstructed images
        N = min(10, videos.shape[1])
        savepath = os.path.join(self.plots_path, cur_dir, f"Recons_{img_idx+1}.png")
        visualize_recons(
                imgs=videos[0, :N].clamp(0, 1),
                recons=recons_history[0, :N].clamp(0, 1),
                n_cols=10,
                savepath=savepath
            )

        # saving the reconstructed objects and masks
        savepath = os.path.join(self.plots_path, cur_dir, f"masks_{img_idx+1}.png")
        _ = visualize_decomp(
                recons_masks[0][:N],
                savepath=savepath,
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
        savepath = os.path.join(self.plots_path, cur_dir, f"maskedObj_{img_idx+1}.png")
        recon_combined = recons_masks[0][:N] * recons_objs[0][:N]
        recon_combined = torch.clamp(recon_combined, min=0, max=1)
        _ = visualize_decomp(
                recon_combined,
                savepath=savepath,
                vmin=0,
                vmax=1,
            )
        return



if __name__ == "__main__":
    utils.clear_cmd()

    # processing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d", "--exp_directory",
            help="Path to the SAVi experiment directory",
            required=True
        )
    parser.add_argument(
            "--savi_ckpt",
            help="Path to SAVi checkpoint to use for figure generation",
            required=True
        )
    parser.add_argument(
            "--num_seqs",
            help="Number of sequences to generate",
            type=int, default=30
        )
    parser.add_argument(
            "--num_frames",
            help="Length (in frames) of the video sequences to process",
            type=int
        )
    parser.add_argument(
            "--set_expert_policy",
            help="If given, expert policy variant is used...",
            default=False, action='store_true'
        )
    args = parser.parse_args()
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)


    print_("Generating figures for SAVI...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            savi_ckpt=args.savi_ckpt,
            num_seqs=args.num_seqs,
            num_frames=args.num_frames,
            set_expert_policy=args.set_expert_policy,
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_savi()
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
