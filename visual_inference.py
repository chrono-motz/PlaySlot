#!/usr/bin/env python3
"""
Visual Inference Script for PlaySlot with LeRobot Actions
This script loads a trained PlaySlot model and generates visual predictions

Usage:
python visual_inference.py -d experiments/ButtonPress --savi_ckpt SAVi_ButtonPress.pth --name_pred_exp PlaySlot --checkpoint PlaySlot_ButtonPress.pth
"""

import argparse
import os
import torch
import numpy as np

# Add src to path
import sys
sys.path.append('src')

from data.load_data import unwrap_batch_data
from lib.config import Config
from lib.logger import print_
import lib.utils as utils
import lib.visualizations as visualizations
import lib.setup_model as setup_model
import data as datalib


class PlaySlotVisualInference:
    """
    Visual inference for PlaySlot model with direct actions
    """
    
    def __init__(self, exp_path, name_pred_exp, savi_ckpt, pred_ckpt):
        """Initialize the visual inference"""
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.pred_ckpt = pred_ckpt
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        self._load_data()
    
    def _load_models(self):
        """Load SAVi and predictor models"""
        print_("Loading SAVi model...")
        # Load SAVi model
        savi_model_path = os.path.join(self.parent_exp_path, "models", self.savi_ckpt)
        savi_checkpoint = torch.load(savi_model_path, map_location=self.device)
        self.savi = setup_model.setup_model(self.exp_params["model"])  # Use "model" not "savi"
        self.savi.load_state_dict(savi_checkpoint["model_state_dict"])
        self.savi.to(self.device)
        self.savi.eval()
        
        print_("Loading predictor model...")
        # Load predictor model - construct the full path
        pred_model_path = os.path.join(self.exp_path, "models", self.pred_ckpt)
        pred_checkpoint = torch.load(pred_model_path, map_location=self.device)
        self.predictor = setup_model.setup_predictor(exp_params=self.exp_params)  # Pass full exp_params
        
        # Handle DataParallel checkpoint - strip 'module.' prefix if present
        state_dict = pred_checkpoint["model_state_dict"]
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        self.predictor.load_state_dict(state_dict)
        self.predictor.to(self.device)
        self.predictor.eval()
        
    def _load_data(self):
        """Load validation dataset"""
        print_("Loading validation dataset...")
        # Override sequence length
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        new_seq_len = num_context + num_preds
        self.exp_params["dataset"]["num_frames"] = new_seq_len
        
        # Load validation set
        self.valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
        print_(f"Loaded {len(self.valid_set)} validation sequences")
        
    @torch.no_grad()
    def predict_sequence(self, imgs, actions, num_context=None, num_preds=None):
        """
        Predict future frames given context frames and actions
        
        Args:
            imgs: Input image sequence [B, T, C, H, W]
            actions: Action sequence [B, T-1, action_dim]
            num_context: Number of context frames
            num_preds: Number of frames to predict
            
        Returns:
            Dictionary with predictions
        """
        if num_context is None:
            num_context = self.exp_params["prediction_params"]["num_context"]
        if num_preds is None:
            num_preds = self.exp_params["prediction_params"]["num_preds"]
            
        B, T, C, H, W = imgs.shape
        num_slots = self.savi.num_slots
        slot_dim = self.savi.slot_dim
        
        # Encode frames to slots using SAVi
        out_model = self.savi(
            imgs,
            num_imgs=num_context + num_preds,
            decode=False
        )
        slot_history = out_model["slot_history"]
        
        # Predict future slots using actions
        pred_slots, pred_others = self.predictor(
            imgs=imgs,
            slots=slot_history,
            actions=actions,
            num_seed=num_context,
            num_preds=num_preds
        )
        
        # Decode predicted slots to images
        pred_slots_dec = pred_slots.clone().reshape(B * (num_context + num_preds - 1), num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.savi.decode(pred_slots_dec)
        pred_imgs = img_recons.view(B, num_context + num_preds - 1, C, H, W)
        
        return {
            "pred_imgs": pred_imgs,
            "pred_objs": pred_recons,
            "pred_masks": pred_masks,
            "pred_slots": pred_slots,
            **pred_others
        }
    
    def visualize_predictions(self, seq_idx=0, save_dir="visual_results"):
        """
        Visualize predictions for a specific sequence
        
        Args:
            seq_idx: Index of sequence to visualize
            save_dir: Directory to save results
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get a sequence from validation set
        imgs, targets, others = self.valid_set[seq_idx]
        actions = others.get("actions", None)
        
        # Add batch dimension
        imgs = imgs.unsqueeze(0).to(self.device)
        if actions is not None:
            actions = actions.unsqueeze(0).to(self.device)
        
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        
        # Get predictions
        print_(f"Generating predictions for sequence {seq_idx}...")
        pred_results = self.predict_sequence(imgs, actions, num_context, num_preds)
        
        # Extract context and target frames
        context_frames = imgs[0, :num_context].cpu()
        target_frames = imgs[0, num_context:num_context+num_preds].cpu()
        pred_frames = pred_results["pred_imgs"][0, num_context-1:num_context+num_preds-1].cpu()
        
        # Clamp values
        context_frames = torch.clamp(context_frames, 0, 1)
        target_frames = torch.clamp(target_frames, 0, 1)
        pred_frames = torch.clamp(pred_frames, 0, 1)
        
        # Create qualitative evaluation figure
        fig, _ = visualizations.visualize_qualitative_eval(
            context=context_frames,
            targets=target_frames,
            preds=pred_frames,
            savepath=os.path.join(save_dir, f"prediction_seq_{seq_idx}.png")
        )
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        # Visualize object decomposition
        pred_objs = pred_results["pred_objs"]
        pred_masks = pred_results["pred_masks"]
        
        # Show predicted objects for the last predicted frame
        last_pred_idx = num_preds - 1
        obj_start_idx = last_pred_idx * self.savi.num_slots
        obj_end_idx = (last_pred_idx + 1) * self.savi.num_slots
        
        last_pred_objs = pred_objs[obj_start_idx:obj_end_idx]
        last_pred_masks = pred_masks[obj_start_idx:obj_end_idx]
        
        fig = visualizations.visualize_decomp(
            (last_pred_objs * last_pred_masks).clamp(0, 1),
            savepath=os.path.join(save_dir, f"objects_seq_{seq_idx}.png"),
            tag=f"Predicted Objects - Sequence {seq_idx}"
        )
        if fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except:
                pass
        
        print_(f"Results saved to {save_dir}/")
        return pred_results
    
    def generate_multiple_predictions(self, num_sequences=5, save_dir="visual_results"):
        """Generate predictions for multiple sequences"""
        print_(f"Generating predictions for {num_sequences} sequences...")
        
        for i in range(min(num_sequences, len(self.valid_set))):
            try:
                print_(f"Processing sequence {i+1}/{num_sequences}")
                self.visualize_predictions(seq_idx=i, save_dir=save_dir)
            except Exception as e:
                print_(f"Error processing sequence {i}: {e}")
                continue
                
        print_(f"Completed! Results saved in {save_dir}/")
    
    def interactive_prediction(self):
        """Interactive prediction mode"""
        print_("Interactive Prediction Mode")
        print_("Enter sequence indices to visualize (or 'quit' to exit)")
        
        while True:
            try:
                user_input = input(f"Enter sequence index (0-{len(self.valid_set)-1}): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                seq_idx = int(user_input)
                if 0 <= seq_idx < len(self.valid_set):
                    self.visualize_predictions(seq_idx, save_dir="interactive_results")
                else:
                    print_(f"Invalid index. Please enter a number between 0 and {len(self.valid_set)-1}")
                    
            except ValueError:
                print_("Please enter a valid number or 'quit'")
            except KeyboardInterrupt:
                break
                
        print_("Exiting interactive mode.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visual Inference for PlaySlot")
    parser.add_argument(
        "-d", "--exp_directory",
        help="Path to the SAVi experiment directory",
        required=True
    )
    parser.add_argument(
        "--savi_ckpt",
        help="Name of the pretrained SAVi checkpoint",
        required=True
    )
    parser.add_argument(
        "--name_pred_exp",
        help="Name of the predictor experiment",
        required=True
    )
    parser.add_argument(
        "--checkpoint",
        help="Name of the predictor checkpoint",
        required=True
    )
    parser.add_argument(
        "--num_sequences",
        help="Number of sequences to visualize",
        type=int,
        default=5
    )
    parser.add_argument(
        "--save_dir",
        help="Directory to save visual results",
        default="visual_results"
    )
    parser.add_argument(
        "--interactive",
        help="Run in interactive mode",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Process arguments like in the training script
    exp_path = utils.process_experiment_directory_argument(args.exp_directory)
    savi_ckpt = utils.process_checkpoint_argument(exp_path, args.savi_ckpt)
    name_pred_exp = utils.process_predictor_experiment(
        exp_directory=exp_path,
        name_predictor_experiment=args.name_pred_exp,
    )
    pred_ckpt = utils.process_predictor_checkpoint(
        exp_path=exp_path,
        name_predictor_experiment=name_pred_exp,
        checkpoint=args.checkpoint
    )
    
    # Initialize visual inference
    visualizer = PlaySlotVisualInference(
        exp_path=exp_path,
        name_pred_exp=name_pred_exp,
        savi_ckpt=savi_ckpt,
        pred_ckpt=pred_ckpt
    )
    
    if args.interactive:
        visualizer.interactive_prediction()
    else:
        visualizer.generate_multiple_predictions(
            num_sequences=args.num_sequences,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    main()
