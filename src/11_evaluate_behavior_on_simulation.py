"""
Evaluating a trained PlaySlot model along with its Policy model
and Action decoder on a simulation environment.
"""

import argparse
import os
os.environ['MUJOCO_GL']='egl'
import torch

from base.baseEvaluator import BaseEvaluator
from base.baseSimulation import BaseSimulationEvaluation
from lib.config import Config
from lib.logger import Logger, print_
import lib.setup_model as setup_model
import lib.utils as utils
import lib.visualizations as visualizations




class Evaluator(BaseEvaluator, BaseSimulationEvaluation):
    """
    Evaluating a trained PlaySlot model along with its Policy model
    and Action decoder on a simulation environment.
    """

    def __init__(
            self, exp_path, name_pred_exp, savi_ckpt, pred_ckpt, name_beh_exp, beh_ckpt,
            seed, num_sims=10, max_num_steps=20, save_vis=10,
        ):
        """ Simple dataset and model checks """
        utils.set_random_seed(seed)
        
        self.savi_exp_path = exp_path
        self.pred_exp_path = os.path.join(exp_path, name_pred_exp)
        self.exp_path = os.path.join(exp_path, name_pred_exp, name_beh_exp)
        self.name_beh_exp = name_beh_exp
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
        self.beh_ckpt = beh_ckpt
        self.set_expert_policy_dataset()
        
        # extra params
        self.seed = seed
        self.num_sims = num_sims
        self.max_num_steps = max_num_steps
        self.save_vis = save_vis
        
        # models paths
        self.savi_models_path = os.path.join(self.savi_exp_path, "models")
        self.pred_models_path = os.path.join(self.pred_exp_path, "models")
        self.models_path = os.path.join(self.exp_path, "models")

        # paths for generating plots and saving results
        exp_name = self._create_exp_name()
        self.plots_path = os.path.join(
                self.exp_path, "plots", "sim_plots", f"SimPlots_{exp_name}"
            )
        utils.create_directory(self.plots_path)
        self.results_path = os.path.join(self.exp_path, "results")
        utils.create_directory(self.results_path)
        self.results_name = f"results_{exp_name}.json"
        return

    def _create_exp_name(self):
        """
        Creating an experiment key with the relevant arguments.
        This will be used for the results and plots directories
        """
        beh_name = self.name_beh_exp.split("/")[-1]
        exp_name = ""
        exp_name = exp_name + f"NumSims={self.num_sims}_"
        exp_name = exp_name + f"NumSteps={self.max_num_steps}_"
        exp_name = exp_name + f"BehaviorExp={beh_name}_"        
        exp_name = exp_name + f"Seed={self.seed}_"        
        
        print_(f"Using Experiment Key:")
        print_(f"  --> {exp_name}")
        return exp_name


    def load_behavior_models(self):
        """
        Instanciating dowsntream models, i.e. Behavior-Cloning and Action Decoding
        """
        # instanciating action decoder
        action_decoder = setup_model.setup_behavior_model(self.exp_params, key="action")
        action_ckp = self.beh_ckpt.replace("Policy", "ActDec")
        checkpoint_path = os.path.join(self.models_path, action_ckp)
        action_decoder = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=action_decoder,
                only_model=True
            )
        self.action_decoder = action_decoder.eval().to(self.device)
        
        # instanciating policy model
        policy_model = setup_model.setup_behavior_model(self.exp_params, key="behavior")
        checkpoint_path = os.path.join(self.models_path, self.beh_ckpt)
        policy_model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=policy_model,
                only_model=True
            )
        self.policy_model = policy_model.eval().to(self.device)
        return
    

    @torch.no_grad()
    def forward_eval(self, idx):
        """ 
        Applying learned models and behaviors in latent imagination and 
        on the simulator
        """
        init_img = self.simulator.init_seq(idx=idx)
        init_img = init_img.to(self.device)
        InvDyn = self.predictor.latent_action
        cOCVP = self.predictor.dynamics_model
        
        # decomposition into slots for first frame
        out_model = self.savi(
                init_img.unsqueeze(1),
                num_imgs=1,
                decode=True
            )
        cur_slots = out_model["slot_history"]
        num_slots = cur_slots.shape[-2]
        recons_init_img = out_model["recons_imgs"][:, 0]
        recons_init_objs = out_model["recons_objs"][:, 0]
        recons_init_masks = out_model["masks"][:, 0]
        
        # representations to track
        all_pred_slots = [cur_slots]
        all_sim_imgs = [init_img]
        all_pred_imgs = [recons_init_img]
        all_pred_objs = [recons_init_objs]
        all_pred_masks = [recons_init_masks]
        action_protos_hist, action_vars_hist = [], []
        
        # autoregressively predicting actions using learned models and behaviors
        for i in range(self.max_num_steps):
            # estimating latent action using behavior and decoding to simulator space
            latent_action = self.policy_model(cur_slots[:, i:i+1])[:, 0]
            action_proto, action_var = InvDyn.decompose_action_latent(latent_action)
            action_protos_hist.append(action_proto)
            action_vars_hist.append(action_var)
            actions = self.action_decoder(latent_action)
            
            # predicting next slots conditioned on latent action
            cur_action_protos = torch.stack(action_protos_hist, dim=1)
            cur_action_vars = torch.stack(action_vars_hist, dim=1)
            cur_slots, cur_action_protos, cur_action_vars = cOCVP.enforce_window(
                    slots=cur_slots,
                    action_protos=cur_action_protos,
                    action_vars=cur_action_vars
                ) 
            if len(cur_action_protos.shape) == 3:  # required for single action model
                cur_action_protos = cur_action_protos.unsqueeze(2)
                cur_action_protos = cur_action_protos.expand(-1, -1, num_slots, -1)
                cur_action_vars = cur_action_vars.unsqueeze(2)
                cur_action_vars = cur_action_vars.expand(-1, -1, num_slots, -1)
            pred_slots = cOCVP.forward_single(
                    slots=cur_slots,
                    action_protos=cur_action_protos,
                    action_vars=cur_action_vars
                )[:, -1:]
            all_pred_slots.append(pred_slots)
            cur_slots = torch.cat(all_pred_slots, dim=1)
            
            # decoding predicted image and object representations
            pred_img, (pred_objs, pred_masks) = self.savi.decode(pred_slots[:, 0])
            all_pred_imgs.append(pred_img)
            all_pred_objs.append(pred_objs)
            all_pred_masks.append(pred_masks)

            # simulating next step given predicted action
            img, done = self.simulator.update(action=actions[0])
            img = img.to(self.device)
            all_sim_imgs.append(img)
            if done:
                break
        
        reps_out = {
            "sim_imgs": torch.stack(all_sim_imgs, dim=1),
            "pred_imgs": torch.stack(all_pred_imgs, dim=1),
            "pred_objs": torch.stack(all_pred_objs, dim=1),
            "pred_masks": torch.stack(all_pred_masks, dim=1),
        }
        return done, reps_out


    @torch.no_grad()
    def save_visualizations(self, data, idx, success):
        """ Saving some simulationa and predictions data """
        cur_dir = f"{idx:03}_success={success}"
        cur_plots_path = os.path.join(self.plots_path, cur_dir)
        utils.create_directory(cur_plots_path)
        
        # predicted and simulated images
        sim_imgs = data.get("sim_imgs")
        pred_imgs = data.get("pred_imgs")
        visualizations.make_gif(
            sim_imgs[0],
            savepath=os.path.join(cur_plots_path, "sim_GIF_frames.gif"),
            n_seed=1000,
            use_border=True
        )
        visualizations.make_gif(
            pred_imgs[0],
            savepath=os.path.join(cur_plots_path, "pred_GIF_frames.gif"),
            n_seed=1,
            use_border=True
        )

        # segmentations and overlay
        pred_masks = data.get("pred_masks")
        pred_masks_categorical = pred_masks[0].argmax(dim=1)
        if len(pred_masks_categorical.shape) == 3:
            pred_masks_categorical = pred_masks_categorical.unsqueeze(1)
        masks_vis = visualizations.masks_to_rgb(x=pred_masks_categorical)[:, 0]
        masks_categorical = visualizations.idx_to_one_hot(x=pred_masks_categorical[:, 0])
        disp_overlay = visualizations.overlay_segmentations(
            sim_imgs[0].cpu().detach(),
            masks_categorical.cpu().detach(),
            colors=visualizations.COLORS,
            alpha=0.6
        )
        visualizations.make_gif(
                masks_vis,
                savepath=os.path.join(cur_plots_path, "masks_GIF_masks.gif"),
                n_seed=1,
                use_border=True
            )
        visualizations.make_gif(
                disp_overlay,
                savepath=os.path.join(cur_plots_path, "overlay_GIF.gif"),
                n_seed=1,
                use_border=True
            )

        return



if __name__ == "__main__":
    utils.clear_cmd()
    
    # processing command line arguments
    parser = argparse.ArgumentParser()
    # base arguments
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
            "--beh_ckpt",
            help="Name of the Behavior Cloning checkpoint to load for evaluation",
            required=True
        )
    # eval and fig generation arguments
    parser.add_argument(
            "--seed",
            help="Random seed for the simulator",
            default=1000,
            type=int, required=True
        )
    parser.add_argument(
            "--num_sims",
            help="Number of simulations to evaluate and genearte figures for",
            default=10, type=int
        )
    parser.add_argument(
            "--max_num_steps",
            help="Maximum duration of the episodes",
            default=20, type=int
        )
    parser.add_argument(
            "--save_vis",
            help="Number of episodes to save visualizations for",
            default=10, type=int
        )
    args = parser.parse_args()
    
    # sanity checks and argument processing
    # base exp checks    
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    args.savi_model = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    # predictor checks
    args.name_pred_exp = utils.process_predictor_experiment(
            exp_directory=exp_path,
            name_predictor_experiment=args.name_pred_exp,    
        )
    pred_exp_path = os.path.join(exp_path, args.name_pred_exp)
    args.pred_ckpt = utils.process_checkpoint_argument(
            exp_path=pred_exp_path,
            checkpoint=args.pred_ckpt
        )
    # behavior checks
    args.name_beh_exp = utils.process_behavior_experiment(
            exp_directory=pred_exp_path,
            name_behavior_experiment=args.name_beh_exp,
        )
    behavior_exp_path = os.path.join(pred_exp_path, args.name_beh_exp)
    assert os.path.exists(behavior_exp_path)
    assert os.path.exists(os.path.join(behavior_exp_path, "models", args.beh_ckpt))
    
    
    logger = Logger(
            exp_path=f"{args.exp_directory}/{args.name_pred_exp}",
            file_name="logs_sim.txt"
        )
    logger.log_info("Starting Simulation Evaluation", message_type="new_exp")
    print_("Initializing Simulation Evaluator...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
        
    args_dict = vars(args)
    evaluator = Evaluator(
            exp_path=args.exp_directory,
            name_pred_exp=args.name_pred_exp,
            savi_ckpt=args.savi_ckpt,
            pred_ckpt=args.pred_ckpt,
            name_beh_exp=args.name_beh_exp,
            beh_ckpt=args.beh_ckpt,
            seed=args.seed,
            num_sims=args.num_sims,
            max_num_steps=args.max_num_steps,
            save_vis=args.save_vis
        )
    print_("Loading dataset...")
    evaluator.setup_simulation()
    print_("Setting up SAVi...")
    evaluator.load_savi(models_path=evaluator.savi_models_path)
    print_("Setting up Predictor...")
    evaluator.load_predictor(models_path=evaluator.pred_models_path)    
    print_("Setting up Policy Model and Action Decoder...")
    evaluator.load_behavior_models()
    print_("Starting Evaluation...")
    evaluator.evaluate_simulation()

