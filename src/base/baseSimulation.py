""" 
Base module for Simulator-based evaluations and figure
generation procedures
"""

import os
os.environ['MUJOCO_GL']='egl'
import json
from tqdm import tqdm
import torch

from lib.utils import timestamp, create_directory
from lib.logger import print_


class BaseSimulationEvaluation:
    """ 
    Base class used to abstract some functionalities of the Simulator-based
    evaluations and figure generation procedures.
    """


    def setup_simulation(self):
        """ Initializing simulation """
        dataset_name = self.exp_params["dataset"]["dataset_name"]
        if "ButtonPress" in dataset_name:
            from data.ButtonPress_Sim import ButtonPressSim
            self.simulator = ButtonPressSim(
                    num_rand_actions=12,
                    seed=self.seed
                )
        elif "BlockPush" in dataset_name:
            from data.BlockPush_Sim import BlockPushSim
            task_name = "PushOdd_3Distractors_Dense-v1"
            self.simulator = BlockPushSim(task_name, seed=self.seed)
        else:
            raise NameError(f"Upsi... {dataset_name = } not recognized...")
        return


    @torch.no_grad()
    def evaluate_simulation(self):
        """
        Evaluating model epoch loop
        """
        self.successes = []
        progress_bar = tqdm(range(self.num_sims))
        for i in progress_bar:
            done, reps_out = self.forward_eval(idx=i)
            self.successes.append(done)
            if i < self.save_vis:
                self.save_visualizations(reps_out, idx=i, success=done)
            progress_bar.set_description(f"Iter {i}/{self.num_sims}")
        self.aggregate_and_save_simulation_metrics()
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
        results["meta"]["predictor_model"] = self.pred_ckpt
        results["meta"]["behavior_exp"] = self.name_beh_exp
        results["meta"]["behavior_ckpt"] = self.beh_ckpt
        results["meta"]["timestamp"] = timestamp()

        create_directory(self.results_path)
        results_file = os.path.join(self.results_path, fname)
        with open(results_file, "w") as f:
            json.dump(results, f)
        return
