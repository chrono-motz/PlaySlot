"""
Base predictor trainer from which all predictor trainer classes inherit.

Basically it removes the scaffolding that is repeat across all predictor
training modules
"""

import os
from tqdm import tqdm
import torch

from lib.callbacks import Callbacks
from lib.config import Config
from lib.logger import print_, log_function, log_info
from lib.loss import LossTracker
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib
from models.BlocksUtils.model_utils import freeze_params


class BasePredictorTrainer:
    """
    Base Class for training and validating a predictor model

    Args:
    -----
    exp_path: string
        Path to the SAVi experiment directory from which to read its exp. params,
    name_predictor_experiment: string
        Name of the predictor experiment (subdirectory in SAVi directory) to train.
    savi_ckpt: string
        Name of the pretrained SAVI model used to extract object representation
        from frames and to decode the predicted slots back to images
    checkpoint: string/None
        Name of a model checkpoint stored in the models/ directory of the predictor
        experiment directory.
    resume_training: bool
        If True, saved checkpoint states from the optimizer, scheduler, ... are
        restored in order to continue training from the checkpoint
    """

    @log_function
    def __init__(self, name_pred_exp, exp_path, savi_ckpt,
                 checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        self.name_pred_exp = name_pred_exp
        self.savi_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_pred_exp)
        if not os.path.exists(self.exp_path):
            raise FileNotFoundError(f"Predictor {self.exp_path = } does not exist...")
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_ckpt = savi_ckpt
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)

        self.training_losses = []
        self.validation_losses = []
        tboard_logs = os.path.join(
                self.exp_path,
                "tboard_logs",
                f"tboard_{utils.timestamp()}"
            )
        utils.create_directory(tboard_logs)
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
            
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    @log_function
    def load_data(self):
        """
        Loading train and validation datasets and fitting data-loader
        for iterating in batches
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        
        # overriding sequence length of dataset with num_context + num_preds
        num_context = self.exp_params["prediction_params"]["num_context"]
        num_preds = self.exp_params["prediction_params"]["num_preds"]
        new_seq_len = num_context + num_preds
        print_(f"Replacing sequence length with required seq. length of {new_seq_len}")
        self.exp_params["dataset"]["num_frames"] = new_seq_len
            
        train_set = datalib.load_data(exp_params=self.exp_params, split="train")
        print_(f"Examples in training set: {len(train_set)}")
        valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
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

    @log_function
    def load_savi(self, models_path=None):
        """
        Load pretraiened SAVi model from checkpoint
        """
        if models_path is None:
            models_path = os.path.join(self.savi_exp_path, "models")
        # seting up savi
        savi = setup_model.setup_model(model_params=self.exp_params["model"])

        # loading pretrained parameters and freezing SAVi modules
        checkpoint_path = os.path.join(models_path, self.savi_ckpt)
        print_("Loading pretrained model:")
        print_(f"  --> Loading SAVi pretrained params from ckpt {self.savi_ckpt}...")
        savi = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=savi,
                only_model=True
            )
        self.savi = savi.to(self.device).eval()
        freeze_params(self.savi)
        return

    @log_function
    def setup_predictor(self):
        """
        Initializing predictor, optimizer, loss function and other related objects
        """
        # instanciating predictor model
        predictor = setup_model.setup_predictor(exp_params=self.exp_params)
        predictor = predictor.eval().to(self.device)

        # loading optimizer, scheduler and loss
        optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
                exp_params=self.exp_params,
                model=predictor
            )
        loss_tracker = LossTracker(loss_params=self.exp_params["predictor_loss"])
        epoch = 0

        # loading pretrained model and other objects
        if self.checkpoint is not None:
            print_(f"  --> Loading pretrained params from ckpt {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=predictor,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                predictor, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"  --> Resuming training from epoch {epoch}...")
            else:
                predictor = loaded_objects

        self.predictor = predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.loss_tracker = loss_tracker
        self.lr_warmup = lr_warmup

        # logging models
        utils.log_architecture(
                self.savi,
                exp_path=self.exp_path,
                fname="architecture_savi.txt"
            )
        utils.log_architecture(
                self.predictor,
                exp_path=self.exp_path,
                fname="architecture_predictor.txt"
            )

        # setting up callbacks
        self.callback_manager = Callbacks(trainer=self)
        self.callback_manager.initialize_callbacks(trainer=self)
        return

    @emergency_save
    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the
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
            self.predictor.eval()
            self.valid_epoch(epoch)
            self.predictor.train()
            self.train_epoch(epoch)

            # on epoch end callbacks
            callback_returns = self.callback_manager.on_epoch_end(trainer=self)
            stop_training = callback_returns.get("stop_training", False)
            if stop_training:
                break

            # saving predictor checkpoint if reached saving frequency
            self.wrapper_save_checkpoint(epoch=epoch, savename="checkpoint_last_saved.pth")
            if(epoch % save_frequency == 0 and epoch != 0):
                print_("Saving model checkpoint")
                self.wrapper_save_checkpoint(epoch=epoch, savedir="models")

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        self.wrapper_save_checkpoint(epoch=epoch, finished=True)
        return

    def wrapper_save_checkpoint(self, epoch=None, savedir="models",
                                savename=None, finished=False):
        """
        Wrapper for saving a models in a more convenient manner
        """
        setup_model.save_checkpoint(
                model=self.predictor,
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

            # forward pass
            _, loss = self.forward_loss_metric(
                    batch_data=data,
                    training=True
                )

            # logging and visualizations and other values
            if(self.iter_ % self.exp_params["training"]["image_log_frequency"] == 0):
                batch_data = next(iter(self.valid_loader))
                self.predictor.eval()
                self.visualizations(
                    batch_data=batch_data,
                    epoch=epoch,
                    iter_=self.iter_
                )
                self.predictor.train()

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
            _ = self.forward_loss_metric(
                    batch_data=data,
                    training=False
                )
            loss = self.loss_tracker.get_last_losses(total_only=True)
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")
        self.callback_manager.on_valid_epoch_end(trainer=self)
        return


    def optimization(self, loss, iter_, epoch):
        """
        Performing the optimziation of the models. Including backward pass and
        optimization steps.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if self.exp_params["training"]["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(),
                    self.exp_params["training"]["clipping_max_value"]
                )
        self.optimizer.step()
        return


    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
        """
        Computing a forwad pass through the model, and (if necessary) the loss values and metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")

    def visualizations(self):
        """
        Making a visualization of some ground-truth, targets and predictions from the current model.

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        """
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")