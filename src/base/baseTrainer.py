"""
Base trainer from which all backbone trainer classes inherit.
Basically it removes the scaffolding that is repeat across all training modules
"""

import os
from tqdm import tqdm
import torch

from lib.callbacks import Callbacks
from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.loss import LossTracker
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib


@for_all_methods(log_function)
class BaseTrainer:
    """
    Base Class for training and validating a backbone model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment params,
        and where to store logs, plots and checkpoints
    checkpoint: string/None
        Name of a model checkpoint stored in the models folder of the exp directory.
        If given, the model is initialized with the parameters of such checkpoint.
        This can be used to continue training or for transfer learning.
    resume_training: bool
        If True, saved checkpoint states from the optimizer, scheduler, ... are
        restored in order to continue training from the checkpoint
    """

    def __init__(self, exp_path, checkpoint=None, resume_training=False, tboard=True):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        if tboard:
            tboard_logs = os.path.join(
                    self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}"
                )
            utils.create_directory(tboard_logs)
            self.writer = utils.TensorboardWriter(logdir=tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        train_set = datalib.load_data(
                exp_params=self.exp_params,
                split="train"
            )
        print_(f"Examples in training set: {len(train_set)}")
        valid_set = datalib.load_data(
                exp_params=self.exp_params,
                split="valid"
            )
        print_(f"Examples in validation set: {len(valid_set)}")
        self.train_loader = datalib.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.valid_loader = datalib.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_(f"Using {torch.cuda.device_count()} GPUs")
        print_(f"  --> device: {self.device}")

        # loading model
        model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)

        # loading optimizer, scheduler and loss
        optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
                exp_params=self.exp_params,
                model=model
            )
        loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        epoch = 0

        # loading pretrained model and other necessary objects for resuming training
        if self.checkpoint is not None:
            print_(f"Loading pretrained params from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                model, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects

        self.model = model
        self.optimizer, self.scheduler, self.epoch = optimizer, scheduler, epoch
        self.loss_tracker = loss_tracker
        self.lr_warmup = lr_warmup

        # setting up callbacks
        self.callback_manager = Callbacks(trainer=self)
        self.callback_manager.initialize_callbacks(trainer=self)
        return

    @emergency_save
    def training_loop(self):
        """
        Repeating the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.callback_manager.on_epoch_start(trainer=self)

            # validation and training for the current epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)

            # on epoch end callbacks
            callback_returns = self.callback_manager.on_epoch_end(trainer=self)
            stop_training = callback_returns.get("stop_training", False)
            if stop_training:
                break

            # saving backup model ckpt and, if reached saving frequency
            self.wrapper_save_checkpoint(epoch=epoch, savename="checkpoint_last_saved.pth")
            if(epoch % save_frequency == 0 and epoch != 0):  # checkpoint_epoch_xx.pth
                print_("Saving model checkpoint")
                self.wrapper_save_checkpoint(epoch=epoch, savedir="models")

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        self.wrapper_save_checkpoint(epoch=epoch, finished=not stop_training)
        return

    def wrapper_save_checkpoint(self, epoch=None, savedir="models",
                                savename=None, finished=False):
        """
        Wrapper for saving a models in a more convenient manner
        """
        setup_model.save_checkpoint(
                model=self.model.module,
                optimizer=self.optimizer,
                scheduler=self.warmup_scheduler.scheduler,
                lr_warmup=self.warmup_scheduler.lr_warmup,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir=savedir,
                savename=savename,
                finished=finished
            )
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        self.loss_tracker.reset()
        max_train_iters = self.exp_params["training"].get(
                "train_iters_per_epoch", len(self.train_loader)
            )
        total_progress_bar = min(len(self.train_loader), max_train_iters)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_progress_bar)
        
        for i, data in progress_bar:
            if i >= max_train_iters:
                break
            self.iter_ = total_progress_bar * epoch + i
            self.callback_manager.on_batch_start(trainer=self)

            # forward pass, computing loss, backward pass, and update step
            _, loss = self.forward_loss_metric(batch_data=data, training=True)

            # logging and visualizations and other values
            if(self.iter_ % self.exp_params["training"]["image_log_frequency"] == 0):
                batch_data = next(iter(self.valid_loader))
                self.visualizations(batch_data=batch_data, iter_=self.iter_)

            # on batch end callbacks
            self.callback_manager.on_batch_end(trainer=self)
            self.callback_manager.on_log_frequency(trainer=self)
            self.callback_manager.on_image_log_frequency(trainer=self)

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")
        self.callback_manager.on_train_epoch_end(trainer=self)
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        self.loss_tracker.reset()
        max_valid_iters = self.exp_params["training"].get(
                "valid_iters_per_epoch", len(self.valid_loader)
            )
        total_progress_bar = min(len(self.valid_loader), max_valid_iters)
        progress_bar = tqdm(enumerate(self.valid_loader), total=total_progress_bar)

        for i, data in progress_bar:
            if i >= max_valid_iters:
                break
            _ = self.forward_loss_metric(batch_data=data, training=False)
            loss = self.loss_tracker.get_last_losses(total_only=True)
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")
        self.callback_manager.on_valid_epoch_end(trainer=self)
        return


    def forward_loss_metric(self, batch_data, training=False, inference_only=False):
        """
        Computing a forwad pass through the model, and
        (if necessary) the loss values and optimziation

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, actions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed.
            Useful for generating images.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        raise NotImplementedError("BaseTrainer does not implement 'forward_loss_metric'...")


    def visualizations(self, batch_data, iter_):
        """
        Making a visualization of some GT, targets and predictions

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch,
            including images, or metadata, among others.
        iter_: int
            Number of the current training iteration.
        """
        raise NotImplementedError("BaseTrainer does not implement 'visualizations'...")


#
