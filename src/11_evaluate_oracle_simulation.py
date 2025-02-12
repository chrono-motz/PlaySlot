"""
Evaluating a Behavior Oracle to complete a task by exectuing 
the learned behavior from object-centric representations. 
"""

import argparse
import os
import json
os.environ['MUJOCO_GL']='egl'
import torch

from base.baseEvaluator import BaseEvaluator
from base.baseSimulation import BaseSimulationEvaluation
from lib.logger import Logger, print_
from lib.config import Config
import lib.setup_model as setup_model
import lib.utils as utils
import lib.visualizations as vis



class Evaluator(BaseEvaluator, BaseSimulationEvaluation):
    """
    Training and validating an Action-Decoder module in order to map the
    learned Latent Actions into the action space of the robot/simulation
    """

    def __init__(self, exp_path, savi_ckpt, name_oracle_exp, oracle_ckpt,
                seed, num_sims=10, max_num_steps=20, save_vis=10):
        """ Simple dataset and model checks """
        self.savi_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_oracle_exp)
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError(f"Downstream {self.exp_path = } does not exist...")
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.oracle_ckpt_name = oracle_ckpt
        self.oracle_ckpt_path = os.path.join(self.exp_path, "models", oracle_ckpt)

        # other sim and eval args
        self.seed = seed
        self.num_sims = num_sims
        self.max_num_steps = max_num_steps
        self.save_vis = save_vis

        # results and visualziations
        exp_name = self._create_exp_name()
        self.models_path = os.path.join(self.exp_path, "models")
        self.savi_models_path = os.path.join(self.savi_exp_path, "models")
        utils.create_directory(self.models_path)
        self.plots_path = os.path.join(
                self.exp_path, "plots", "sim_plots", f"SimPlots_{exp_name}"
            )
        utils.create_directory(self.plots_path)
        self.results_path = os.path.join(self.exp_path, "results", "sim_results")
        utils.create_directory(self.results_path)
        self.results_name = f"results_{exp_name}.json"
        
        # self.predictor = torch.nn.Identity()
        return

    def _create_exp_name(self):
        """
        Creating an experiment key with the relevant arguments.
        This will be used for the results and plots directories
        """
        exp_name = ""
        exp_name = exp_name + f"NumSims={self.num_sims}_"
        exp_name = exp_name + f"NumSteps={self.max_num_steps}_"
        exp_name = exp_name + f"Ckpt={self.oracle_ckpt_name.split('.')[0]}_"
        exp_name = exp_name + f"Seed={self.seed}_"

        print_(f"Using Experiment Key:")
        print_(f"  --> {exp_name}")
        return exp_name

  
    def setup_oracle(self):
        """
        Instanciating Oracle model
        """
        oracle = setup_model.setup_behavior_model(self.exp_params, key="behavior")
        print_(f"  --> Loading pretrained Oracle from {self.oracle_ckpt_path = }...")
        oracle_model = setup_model.load_checkpoint(
                checkpoint_path=self.oracle_ckpt_path,
                model=oracle,
                only_model=True,
            )
        self.oracle_model = oracle_model.eval().to(self.device)
        return
    

    @torch.no_grad()
    def forward_eval(self, idx):
        """ 
        Applying learned models and behaviors on the simulator
        """
        cur_img = self.simulator.init_seq(idx=idx)
        cur_img = cur_img.to(self.device)
        
        all_sim_imgs = [cur_img]
        all_pred_imgs, all_pred_objs, all_pred_masks = [], [], []
        for _ in range(self.max_num_steps):
            # parsing simulated image into slots
            out_model = self.savi(
                    cur_img.unsqueeze(0),
                    num_imgs=1,
                    decode=False
                )
            slots = out_model["slot_history"][:, -1]
            # predicting next action using oracle model
            pred_action = self.oracle_model(slots.unsqueeze(1))
            # rendering frame: for visualization purposes only
            pred_img, (pred_objs, pred_masks) = self.savi.decode(slots)
            all_pred_imgs.append(pred_img)
            all_pred_objs.append(pred_objs)
            all_pred_masks.append(pred_masks)

            # simulating next step given predicted action
            cur_img, done = self.simulator.update(action=pred_action[0])
            cur_img = cur_img.to(self.device)
            all_sim_imgs.append(cur_img)
            if done:
                break

        reps_out = {
                "sim_imgs": torch.stack(all_sim_imgs, dim=1),
                "pred_imgs": torch.stack(all_pred_imgs, dim=1),
                "pred_objs": torch.stack(all_pred_objs, dim=1),
                "pred_masks": torch.stack(all_pred_masks, dim=1)
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
        vis.make_gif(
            sim_imgs[0],
            savepath=os.path.join(cur_plots_path, "sim_GIF_frames.gif"),
            n_seed=1000,
            use_border=True
        )
        vis.make_gif(
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
        masks_vis = vis.masks_to_rgb(x=pred_masks_categorical)[:, 0]
        masks_categorical = vis.idx_to_one_hot(x=pred_masks_categorical[:, 0])
        disp_overlay = vis.overlay_segmentations(
                sim_imgs[0].cpu().detach(),
                masks_categorical.cpu().detach(),
                colors=vis.COLORS,
                alpha=0.6
            )
        vis.make_gif(
                masks_vis,
                savepath=os.path.join(cur_plots_path, "masks_GIF_masks.gif"),
                n_seed=1,
                use_border=True
            )
        vis.make_gif(
                disp_overlay,
                savepath=os.path.join(cur_plots_path, "overlay_GIF.gif"),
                n_seed=1,
                use_border=True
            )
        return


    @torch.no_grad()
    def aggregate_and_save_simulation_metrics(self, fname=None):
        """
        Aggregating all computed metrics and saving results to logs file
        """
        fname = self.results_name if fname is None else fname
        success_rate = sum(self.successes) / len(self.successes)
        print_(f"Success Rate: {round(success_rate * 100, 1)}%")
        results = {
            "_results": {},
            "meta": {},
        }
        results["_results"] = {
            "success_rate": success_rate,
            "per_seq_succsess": {i: s for i, s in enumerate(self.successes)}
        }
        # adding more metadata to the results file
        results["meta"]["savi_model"] = self.savi_ckpt
        results["meta"]["oracle_ckpt"] = self.oracle_ckpt_name
        results["meta"]["timestamp"] = utils.timestamp()

        results_file = os.path.join(self.results_path, fname)
        with open(results_file, "w") as file:
            json.dump(results, file)
        return



if __name__ == "__main__":
    utils.clear_cmd()
    
    # command line arguments
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
            "--oracle_ckpt",
            help="Checkpoint of the pretrained Oracle",
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

    # checks
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    args.savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    args.name_oracle_exp = "oracle/" + args.name_oracle_exp
    if not os.path.exists(os.path.join(exp_path, args.name_oracle_exp)):
        raise FileNotFoundError(F"Oracle exp {args.name_oracle_exp} does not exist...")    
    args.oracle_ckpt = utils.process_checkpoint_argument(
            os.path.join(exp_path, args.name_oracle_exp),
            args.oracle_ckpt
        )
    
    # Evaluating Oracle and generating figures
    logger = Logger(
            exp_path=f"{args.exp_directory}/{args.name_oracle_exp}",
            file_name="logs_sim.txt"
        )
    logger.log_info("Starting Simulation Evaluation for Oracle", message_type="new_exp")

    print_("Initializing Simulation Evaluator for Oracle...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
        
    args_dict = vars(args)
    evaluator = Evaluator(
            exp_path=args.exp_directory,
            savi_ckpt=args.savi_ckpt,
            name_oracle_exp=args.name_oracle_exp,
            oracle_ckpt=args.oracle_ckpt,
            seed=args.seed,
            num_sims=args.num_sims,
            max_num_steps=args.max_num_steps,
            save_vis=args.save_vis
        )

    print_("Setting up SAVi...")
    evaluator.load_savi(models_path=evaluator.savi_models_path)
    print_("Setting up Oracle...")
    evaluator.setup_oracle()
    print_("Loading dataset...")
    evaluator.setup_simulation()
    print_("Starting Evaluation...")
    evaluator.evaluate_simulation()


#
