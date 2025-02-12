"""
Default callbacks file.
This includes nice goodies to have during training including:
  - Logging losses to tensorboard
  - Orquestrating Learning-Rate warmup and scheduling
  - Saving train/valid loss values to a json file, and saving some loss plots
"""

import os
import json
from matplotlib import pyplot as plt

from lib.callbacks import Callback
from lib.schedulers import WarmupVSScehdule
from lib.utils import create_directory



class LogTensorboard(Callback):
    """
    Logging training and validation losses onto tensorboard
    """

    def on_train_epoch_end(self, trainer):
        """ Logging train losses at the end of every train epoch"""
        trainer.loss_tracker.aggregate()
        average_loss_vals = trainer.loss_tracker.summary(
                log=True,
                get_results=True
            )
        trainer.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=trainer.epoch + 1,
                plot_name="Train Loss",
                dir="Train Loss",
            )
        trainer.training_losses.append(average_loss_vals["_total"].item())
        return


    def on_valid_epoch_end(self, trainer):
        """ Logging valid losses at the end of every valid epoch"""
        trainer.loss_tracker.aggregate()
        average_loss_vals = trainer.loss_tracker.summary(
                log=True,
                get_results=True
            )
        trainer.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=trainer.epoch + 1,
                plot_name="Valid Loss",
                dir="Valid Loss",
            )
        trainer.validation_losses.append(average_loss_vals["_total"].item())


    def on_epoch_end(self, trainer):
        """
        Logging training and validation loss at the end of every epoch
        """
        trainer.writer.add_scalars(
                plot_name='Total Loss',
                val_names=["train_loss", "eval_loss"],
                vals=[trainer.training_losses[-1], trainer.validation_losses[-1]],
                step=trainer.epoch+1
            )

    def on_log_frequency(self, trainer):
        """ Logging losses and learning rate every few iterations """
        trainer.writer.log_full_dictionary(
                dict=trainer.loss_tracker.get_last_losses(),
                step=trainer.iter_,
                plot_name="Train Loss",
                dir="Train Loss Iter",
            )
        trainer.writer.add_scalar(
                name="Learning/Learning Rate",
                val=trainer.optimizer.param_groups[0]['lr'],
                step=trainer.iter_
            )



class WarmupScheduleCallback(Callback):
    """
    Warmup vs Schedule callback
    """

    def __init__(self, trainer, optimizer=None, lr_warmup=None, scheduler=None,
                 warmup_scheduler_name="warmup_scheduler"):
        """ Callback initializer """
        # fetching objects
        def get_attr(name, x):
            return x if x is not None else getattr(trainer, name, None)
        optimizer = get_attr(name="optimizer", x=optimizer)
        scheduler = get_attr(name="scheduler", x=scheduler)
        lr_warmup = get_attr(name="lr_warmup", x=lr_warmup)

        # instanciating warmup-scheduler
        if lr_warmup is None or scheduler is None:
            warmup_scheduler = None
        else:    
            warmup_scheduler = WarmupVSScehdule(
                    optimizer=optimizer,
                    lr_warmup=lr_warmup,
                    scheduler=scheduler
                )
        setattr(trainer, warmup_scheduler_name, warmup_scheduler)
        self.warmup_scheduler_name = warmup_scheduler_name
        return

    def on_batch_start(self, trainer):
        """ """
        warmup_scheduler = getattr(trainer, self.warmup_scheduler_name)
        if warmup_scheduler is not None:
            warmup_scheduler(
                    iter=trainer.iter_,
                    epoch=trainer.epoch,
                    exp_params=trainer.exp_params,
                    end_epoch=False
                )
        return

    def on_epoch_end(self, trainer):
        """ Updating at the end of every epoch """
        warmup_scheduler = getattr(trainer, self.warmup_scheduler_name)
        if warmup_scheduler is not None:
            warmup_scheduler(
                    iter=-1,
                    epoch=trainer.epoch,
                    exp_params=trainer.exp_params,
                    end_epoch=True
                )
        return



class LogLossesToJSON(Callback):
    """
    Logging losses into a JSON file to later make loss plots without having to load
    the entire tensorboard, which is often slow due to images.
    """

    def __init__(self, trainer):
        """ Callback initializer and loading previous losses if resuming training """
        self.loss_file = os.path.join(trainer.exp_path, "losses.json")
        self.plots_path = os.path.join(trainer.exp_path, "plots", "loss_plots")
        create_directory(self.plots_path)
        
        if trainer.resume_training and os.path.exists(self.loss_file):
            with open(self.loss_file, "r") as f:
                self.losses = json.load(f)
        else:
            self.losses = {}
        return

    def _get_losses(self, trainer, is_train=False):
        """ fetching and adding new losses to the dictionary """
        label = "train" if is_train else "valid"
        last_losses = trainer.loss_tracker.get_last_losses(total_only=False)
        for loss_name, loss_value in last_losses.items():
            if loss_name not in self.losses.keys():
                self.losses[loss_name] = {}
                self.losses[loss_name]["train"] = []
                self.losses[loss_name]["valid"] = []
            self.losses[loss_name][label].append(loss_value.cpu().item())
        return
    
    def on_train_epoch_end(self, trainer):
        """ Logging train losses at the end of every train epoch"""
        self._get_losses(trainer, is_train=True)
        return

    def on_valid_epoch_end(self, trainer):
        """ Logging vaöid losses at the end of every valid epoch"""
        self._get_losses(trainer, is_train=False)

    def on_epoch_end(self, trainer):
        """ We only store the losses at the end of an epoch """        
        # saving the updated losses dictionary to the file
        with open(self.loss_file, "w") as f:
            json.dump(self.losses, f)
            
        # plotting the losses
        for loss_name in self.losses.keys():
            self.make_plot_loss(
                    train_losses=self.losses[loss_name]["train"],
                    val_losses=self.losses[loss_name]["valid"],
                    loss_name=loss_name
                )
        return
    
    def make_plot_loss(self, train_losses, val_losses, loss_name):
        """ Making a plot for a loss """
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 4)
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="valid")
        ax.set_title(f"{loss_name} Loss")
        ax.legend()
        plt.savefig(os.path.join(self.plots_path, f"{loss_name}.png")) 
        ax.set_yscale("log")
        plt.savefig(os.path.join(self.plots_path, f"{loss_name}_log.png")) 
        return
    
    
