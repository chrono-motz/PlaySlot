"""
Generating some figures using a pretrained PlaySlot model by repeatedly
conditioning the prediction process on the same learned action prototype.
"""

import argparse
import os
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from base.baseEvaluator import BaseEvaluator
from data.load_data import unwrap_batch_data
from lib.config import Config
from lib.logger import print_
import lib.utils as utils
import lib.visualizations as vis



class FigGenerator(BaseEvaluator):
    """
    Generating some figures using a pretrained PlaySlot model by repeatedly
    conditioning the prediction process on the same learned action prototype.
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
                f"ActionFigs_Pred_{self.pred_name}_{name_pred_exp.split('/')[-1]}_" + \
                f"{pred_ckpt[:-4]}_" + \
                f"NumSeed={num_seed}_NumPreds={num_preds}_" + \
                f"_ExpertPolicy={set_expert_policy}"
        self.plots_path = os.path.join(self.exp_path, "plots", self.plots_dir)
        utils.create_directory(self.plots_path)


    @torch.no_grad()
    def generate_figs(self):
        """
        Iterating over sequences and generating figures
        """
        utils.set_random_seed()
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        
        # fetching latent action model
        if hasattr(self.predictor, "latent_action"):
            inv_dyn = self.predictor.latent_action
        else:
            raise NameError(f"'latent_action' module could not be found in predictor")
        self.inv_dyn = inv_dyn

        for idx in tqdm(range(self.num_seqs)):
            # preparing data and sanity checks
            batch_data = self.test_set[idx]
            videos, _, init_data, _ = unwrap_batch_data(self.exp_params, batch_data)
            videos = videos.unsqueeze(0).to(self.device)
            init_data = {
                    k: v.unsqueeze(0) for k, v in init_data.items() if torch.is_tensor(v)
                }
            n_frames = videos.shape[1]
            if n_frames < num_context + num_preds:
                raise ValueError(f"Seq-len {n_frames} < {num_context + num_preds = }")

            # visualizing pairwise distance between codewords
            codewords = self.inv_dyn.quantizer.get_codewords()
            _ = vis.visualize_distance_between_centroids(codewords)
            plt.savefig(os.path.join(self.plots_path, f"codeword_dist.png"))

            # forward pass through object-centric prediction model
            all_pred_imgs, titles = self.forward_pass(
                    videos=videos,
                    init_data=init_data
                )

            # generating and saving visualizations
            self.compute_visualization(
                    videos=videos,
                    pred_imgs=all_pred_imgs,
                    titles=titles,
                    seq_idx=idx,
                    img_prefix="preds"
                )
        return


    @torch.no_grad()
    def forward_pass(self, videos, init_data):
        """
        Forward pass through SAVi and Preditor
        """
        B, L = videos.shape[0], videos.shape[1]
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        use_variability = self.predictor.use_variability
        all_pred_imgs, all_titles = [], []

        # computing object slots
        out_model = self.savi(videos, num_imgs=L, **init_data)
        slot_history = out_model["slot_history"]
        seed_slots = slot_history[:, :num_context]
        
        # predictions using latent action inferred from the video
        out_dict = self.inv_dyn.compute_actions(slot_history)
        post_action_proto = out_dict["action_proto"]
        post_varibility = out_dict["action_variability"]
        pred_slots, _ = self.predictor.autoregressive_inference(
                seed_slots,
                action_protos=post_action_proto,
                action_vars=post_varibility if use_variability else None,
                N=num_preds
            )
        pred_imgs = self.savi.decode(pred_slots[0])[0].clamp(0, 1)
        all_pred_imgs.append(pred_imgs)
        all_titles.append("Posterior Actions")

        # predicting future frames using each individual action prototype
        num_actions = self.predictor.num_actions
        action_dim = self.predictor.action_dim
        no_variability_emb = torch.zeros(
                (*post_action_proto.shape[:-1], action_dim)
            ).to(self.device)
        for action_idx in range(num_actions):
            B, num_slots = seed_slots.shape[0], seed_slots.shape[-2]
            single_action_proto, _ = self.inv_dyn.get_action(
                    action_idx,
                    shape=(B, num_context + num_preds, 1, 1)
                )
            single_action_proto = single_action_proto.repeat(1, 1, num_slots, 1)
            pred_slots, _ = self.predictor.autoregressive_inference(
                    seed_slots=seed_slots,
                    action_protos=single_action_proto,
                    action_vars=no_variability_emb if use_variability else None,
                    N=num_preds
                )
            pred_imgs = self.savi.decode(pred_slots[0])[0].clamp(0, 1)
            all_pred_imgs.append(pred_imgs)
            all_titles.append(f"Action {action_idx+1}")

        all_pred_imgs = torch.stack(all_pred_imgs, dim=0)
        return all_pred_imgs, all_titles


    def compute_visualization(self, videos, pred_imgs, titles, seq_idx,
                              img_prefix="preds", **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        videos: torch Tensor
            Videos sequence from the dataset, containing the seed and target frames.
            Shape is (B, num_frames, C, H, W)
        pred_imgs: torch Tensor
            Predicted video frames. Shape is (B, num_preds, C, H, W)
        img_idx: int
            Index of the visualization to compute and save
        """
        cur_dir = kwargs.get("cur_dir", f"img_{seq_idx+1}")
        utils.create_directory(self.plots_path, cur_dir)

        # some hpyer-parameters of the video model
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        seed_imgs = videos[0, :num_context, :, :]
        target_imgs = videos[0, num_context:num_context+num_preds, :, :]

        # Video prediction figrue
        _ = vis.visualize_stoch_frame_figs(
            context=seed_imgs,
            targets=target_imgs,
            all_preds=pred_imgs,
            titles=titles,
            n_cols=num_preds,
            fontsize=30,
            savepath=os.path.join(self.plots_path, cur_dir, f"{img_prefix}.png")
        )

        # video prediction GIFs
        duration = int((num_context + num_preds) * 20)
        vis.all_tensors_to_gif(
            seed_frames=seed_imgs,
            target_frames=target_imgs,
            all_preds_frames=pred_imgs,
            fpath=os.path.join(self.plots_path, cur_dir, f"{img_prefix}.gif"),
            duration=duration
        )
        
        # saving GIFs from each action
        for i in range(len(titles)):
            cur_frames = torch.cat([videos[0, :1], pred_imgs[i]], dim=0)
            gif_name = f"inferred_dynamics" if i == 0 else f"action_proto_{i}"
            vis.make_gif(
                cur_frames, n_seed=1, use_border=True,
                savepath=os.path.join(self.plots_path, cur_dir, f"{gif_name}.gif"),
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
