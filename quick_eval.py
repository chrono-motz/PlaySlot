#!/usr/bin/env python3
"""
Quick evaluation script to assess PlaySlot model performance
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
import sys
sys.path.append('src')

from data.load_data import unwrap_batch_data
from lib.config import Config
from lib.logger import print_
from lib.metrics import compute_lpips, compute_psnr, compute_ssim
import lib.utils as utils
import lib.setup_model as setup_model
import data as datalib


class PlaySlotEvaluator:
    """Quick evaluator for PlaySlot model"""
    
    def __init__(self, exp_path, name_pred_exp, savi_ckpt, pred_ckpt):
        """Initialize evaluator"""
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.pred_ckpt = pred_ckpt
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_(f"Using device: {self.device}")
        
        self._load_models()
        self._load_data()
    
    def _load_models(self):
        """Load models"""
        print_("Loading models...")
        # Load SAVi
        savi_model_path = os.path.join(self.parent_exp_path, "models", self.savi_ckpt)
        savi_checkpoint = torch.load(savi_model_path, map_location=self.device)
        self.savi = setup_model.setup_model(self.exp_params["savi"])
        self.savi.load_state_dict(savi_checkpoint["model_state_dict"])
        self.savi.to(self.device)
        self.savi.eval()
        
        # Load predictor
        pred_model_path = os.path.join(self.exp_path, "models", self.pred_ckpt)
        pred_checkpoint = torch.load(pred_model_path, map_location=self.device)
        self.predictor = setup_model.setup_predictor(self.exp_params["predictor"])
        self.predictor.load_state_dict(pred_checkpoint["model_state_dict"])
        self.predictor.to(self.device)
        self.predictor.eval()
        
    def _load_data(self):
        """Load dataset"""
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        self.exp_params["dataset"]["num_frames"] = num_context + num_preds
        
        self.valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
        print_(f"Loaded {len(self.valid_set)} validation sequences")
    
    @torch.no_grad()
    def evaluate_sequence(self, imgs, actions):
        """Evaluate a single sequence"""
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        
        B, T, C, H, W = imgs.shape
        num_slots = self.savi.num_slots
        slot_dim = self.savi.slot_dim
        
        # Encode to slots
        out_model = self.savi(imgs, num_imgs=num_context + num_preds, decode=False)
        slot_history = out_model["slot_history"]
        
        # Predict
        pred_slots, _ = self.predictor(
            imgs=imgs,
            slots=slot_history,
            actions=actions,
            num_seed=num_context,
            num_preds=num_preds
        )
        
        # Decode predictions
        pred_slots_dec = pred_slots.clone().reshape(B * (num_context + num_preds - 1), num_slots, slot_dim)
        img_recons, _ = self.savi.decode(pred_slots_dec)
        pred_imgs = img_recons.view(B, num_context + num_preds - 1, C, H, W)
        
        # Extract predicted frames vs ground truth
        pred_frames = pred_imgs[:, num_context-1:num_context+num_preds-1]
        target_frames = imgs[:, num_context:num_context+num_preds]
        
        return pred_frames, target_frames
    
    def compute_metrics(self, num_sequences=None):
        """Compute evaluation metrics"""
        if num_sequences is None:
            num_sequences = min(50, len(self.valid_set))
        
        print_(f"Evaluating on {num_sequences} sequences...")
        
        all_mse = []
        all_psnr = []
        total_loss = 0.0
        
        for i in tqdm(range(num_sequences)):
            try:
                # Get data
                imgs, _, others = self.valid_set[i]
                actions = others.get("actions", None)
                
                imgs = imgs.unsqueeze(0).to(self.device)
                if actions is not None:
                    actions = actions.unsqueeze(0).to(self.device)
                
                # Get predictions
                pred_frames, target_frames = self.evaluate_sequence(imgs, actions)
                
                # Compute MSE
                mse = torch.nn.functional.mse_loss(pred_frames, target_frames)
                all_mse.append(mse.item())
                
                # Compute PSNR
                psnr = compute_psnr(pred_frames, target_frames)
                all_psnr.append(psnr.item())
                
                total_loss += mse.item()
                
            except Exception as e:
                print_(f"Error evaluating sequence {i}: {e}")
                continue
        
        # Aggregate results
        avg_mse = np.mean(all_mse)
        avg_psnr = np.mean(all_psnr)
        avg_loss = total_loss / len(all_mse)
        
        print_(f"\n=== Evaluation Results ===")
        print_(f"Average MSE: {avg_mse:.6f}")
        print_(f"Average PSNR: {avg_psnr:.2f} dB")
        print_(f"Average Loss: {avg_loss:.6f}")
        print_(f"Number of sequences: {len(all_mse)}")
        
        # Interpret results
        self._interpret_results(avg_mse, avg_psnr, avg_loss)
        
        return {
            "mse": avg_mse,
            "psnr": avg_psnr,
            "loss": avg_loss,
            "num_sequences": len(all_mse)
        }
    
    def _interpret_results(self, mse, psnr, loss):
        """Provide interpretation of results"""
        print_(f"\n=== Result Interpretation ===")
        
        # MSE interpretation
        if mse < 0.01:
            print_(f"✓ Excellent MSE ({mse:.6f}) - Very good reconstruction quality")
        elif mse < 0.05:
            print_(f"✓ Good MSE ({mse:.6f}) - Decent reconstruction quality")
        elif mse < 0.1:
            print_(f"! Fair MSE ({mse:.6f}) - Acceptable but could be better")
        else:
            print_(f"✗ Poor MSE ({mse:.6f}) - Poor reconstruction quality")
        
        # PSNR interpretation
        if psnr > 25:
            print_(f"✓ Excellent PSNR ({psnr:.2f} dB) - High quality predictions")
        elif psnr > 20:
            print_(f"✓ Good PSNR ({psnr:.2f} dB) - Decent quality predictions")
        elif psnr > 15:
            print_(f"! Fair PSNR ({psnr:.2f} dB) - Acceptable quality")
        else:
            print_(f"✗ Poor PSNR ({psnr:.2f} dB) - Low quality predictions")
        
        # Loss interpretation
        if loss < 0.05:
            print_(f"✓ Good training convergence (loss: {loss:.6f})")
        elif loss < 0.1:
            print_(f"! Acceptable convergence (loss: {loss:.6f})")
        else:
            print_(f"✗ Poor convergence (loss: {loss:.6f}) - May need more training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", required=True)
    parser.add_argument("--savi_ckpt", required=True)
    parser.add_argument("--name_pred_exp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_sequences", type=int, default=50)
    
    args = parser.parse_args()
    
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    
    evaluator = PlaySlotEvaluator(
        exp_path=exp_path,
        name_pred_exp=args.name_pred_exp,
        savi_ckpt=args.savi_ckpt,
        pred_ckpt=args.checkpoint
    )
    
    evaluator.compute_metrics(num_sequences=args.num_sequences)


if __name__ == "__main__":
    main()
