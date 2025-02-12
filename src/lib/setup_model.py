"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import traceback
import torch

from lib.logger import print_, log_function
from lib.schedulers import LRWarmUp, IdentityScheduler
from lib.utils import create_directory
from configs import get_available_configs

from models.Predictors.predictor_wrappers import PredictorWrapper



###########################
## MODEL FACTORY METHODS ##
###########################



@log_function
def setup_model(model_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    model_name = model_params["model_name"]
    model_params = model_params["model_params"]
    MODELS = get_available_configs("models")
    if model_name not in MODELS:
        raise NameError(f"'{model_name = }' not in {MODELS = }")

    if(model_name == "SAVi"):
        from models.SAVi import SAVi
        model = SAVi(**model_params)
    else:
        raise NameError(f"'{model_name = }' not in recognized {MODELS = }")

    return model



@log_function
def setup_predictor(exp_params):
    """
    Loading the predictor given the predictor parameters stated in the exp_params file

    Args:
    -----
    predictor_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    predictor: PredictorWrapper
        Instanciated predictor given the parameters, wrapped into a PredictorWrapper to
        forecast slots for future time steps.
    """
    # model and predictor params
    model_params = exp_params["model"]["model_params"]
    predictor_name = exp_params["predictor"]["predictor_name"]
    predictor_params = exp_params["predictor"]["predictor_params"]
    PREDICTORS = get_available_configs("predictors")
    if predictor_name not in PREDICTORS:
        raise NameError(f"Predictor '{predictor_name}' not in {PREDICTORS = }")

    # OCVP PREDICTORS
    if(predictor_name == "VanillaTransformer"):
        from models.Predictors.transformers import VanillaTransformerPredictor
        predictor = VanillaTransformerPredictor(
                num_slots=model_params["num_slots"],
                slot_dim=model_params["slot_dim"],
                input_buffer_size=14,
                **predictor_params
            )
        predictor = PredictorWrapper(exp_params=exp_params, predictor=predictor)
    elif(predictor_name == "OCVPSeq"):
        from models.Predictors.transformers import OCVPSeq
        predictor = OCVPSeq(
                num_slots=model_params["num_slots"],
                slot_dim=model_params["slot_dim"],
                input_buffer_size=14,
                **predictor_params
            )
        predictor = PredictorWrapper(exp_params=exp_params, predictor=predictor)
    # SLOT-BASED ACTION-CONDITIONAL PREDICTORS
    elif(predictor_name == "ActionCondOCVP"):
        from models.Predictors.ActionConditional_OCVP import ActionCondtionalOCVP
        predictor = ActionCondtionalOCVP(**predictor_params)       
    # SLOT-BASED LATENT ACTION PREDICTORS
    elif(predictor_name == "SlotLatentPredictor"):
        from models.Predictors.LatentActionPredictors import SlotLatentPredictor
        predictor = SlotLatentPredictor(**predictor_params)       
    elif(predictor_name == "SlotSingleAction"):
        from models.Predictors.LatentActionPredictors import SlotSingleActionPredictor
        predictor = SlotSingleActionPredictor(**predictor_params)
    else:
        raise NameError(f"Pedictors {predictor_name} not in {PREDICTORS = }")

    return predictor

        
        
@log_function
def setup_behavior_model(exp_params, key="behavior"):
    """
    Loading the Behavior Module given the behavior module parameters stated
    in the exp_params file

    Args:
    -----
    exp_params: dictionary
        experiment parameters
    key: string
        Type of downstream task to solve: ['behavior', 'action']

    Returns:
    --------
    behavior_module: nn.Module
        Instanciated BehaviorPredictor module
    """
    assert key in ["behavior", "action"]
    if "predictor" not in exp_params.keys():  # HACK for Oracle
        exp_params["predictor"] = {}
        exp_params["predictor"]["predictor_params"] = {
                "slot_dim": exp_params["model"]["model_params"]["slot_dim"],
                "action_dim": 4  # target action dim
            }
    predictor_params = exp_params["predictor"]["predictor_params"]

    # Choosing the task that we are about to address
    if key == "behavior":
        dict_key = "behavior_model"
        AVAILABLE_MODELS = get_available_configs("behavior_models")
    elif key == "action":
        dict_key = "action_decoder"
        AVAILABLE_MODELS = get_available_configs("action_decoders")
    else:
        raise NameError(f"{key = } was not recognized...")

    # Checking that behavior/action model parameters are in exp-params
    if dict_key not in exp_params:
        raise KeyError(f"{dict_key = } not in 'exp_params'...")
    downstream_model_name = exp_params[dict_key]["model_name"]
    downstream_params = exp_params[dict_key]["model_params"]
    if downstream_model_name not in AVAILABLE_MODELS:
        raise NameError(f"'{downstream_model_name = }' not in {AVAILABLE_MODELS = }")

    # instanciating model for downstream task
    if(downstream_model_name == "MarkovBehaviorCloner"):
        from models.Downstream.behavior_predictor import MarkovBehaviorCloner
        downstream_model = MarkovBehaviorCloner(
                slot_dim=predictor_params["slot_dim"],
                action_dim=predictor_params["action_dim"],
                **downstream_params
            )
    elif(downstream_model_name == "MLPDecoder"):
        from models.Downstream.action_decoding import MLPActionDecoder
        downstream_model = MLPActionDecoder(**downstream_params)
    else:
        raise NameError(
                f"{downstream_model_name = } for {key = } not in {AVAILABLE_MODELS = }"
            )

    return downstream_model



########################
## SAVING AND LOADING ##
########################
        
        

@log_function
def save_checkpoint(model, optimizer=None, scheduler=None, lr_warmup=None,
                    epoch=0, exp_path="", finished=False,
                    savedir="models", savename=None, model_only=False, prefix=""):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    scheduler: Object
        Learning rate scheduler to save
    lr_warmup: Object
        Module performing learning rate warm-up
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = f"{prefix}{savename}"
    elif(savename is None and finished is True):
        checkpoint_name = f"{prefix}checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"{prefix}checkpoint_epoch_{epoch}.pth"

    create_directory(exp_path, savedir)
    savepath = os.path.join(exp_path, savedir, checkpoint_name)

    if model_only:
        torch.save({
                "model_state_dict": model.state_dict()
            }, savepath)
    else:
        scheduler_data = "" if scheduler is None else scheduler.state_dict()
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "scheduler_state_dict": scheduler_data,
                "lr_warmup": lr_warmup
            }, savepath)

    return



@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=True, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)

    # hack to remove the 'module.' corresponding to nn.DataParallel
    model_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            k = k[7:]
        model_state_dict[k] = v
    # tweeks to avoid issues with SlotPositionalEncoding in predictors
    for k in ["dynamics_model.pos_emb.pe", "predictor.pe.pe"]:
        if k in model_state_dict.keys():
            model_state_dict[k] = model.state_dict()[k]

    # returning only model for eval or transfer learning
    model.load_state_dict(model_state_dict)
    if(only_model):
        return model

    # returning all other necessary objects for resuming traiing
    optimizer = kwargs["optimizer"]
    scheduler = kwargs["scheduler"]
    lr_warmup = kwargs["lr_warmup"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if "lr_warmup" in checkpoint:
        lr_warmup.load_state_dict(checkpoint['lr_warmup'])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, lr_warmup, epoch



def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):

    Note: this does not work for predictors, since it saves the backbone, but not the
          predictor model.
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.warmup_scheduler.scheduler,
                    lr_warmup=self_.warmup_scheduler.lr_warmup,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except




############################
## OPTIMIZATION AND UTILS ##
############################


@log_function
def setup_optimizer(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model

    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """
    lr = exp_params["training"]["lr"]
    scheduler = exp_params["training"]["scheduler"]
    scheduler_steps = exp_params["training"].get("scheduler_steps", 1e6)

    # optimizer
    print_("Setting up Adam optimizer:")
    print_(f"    LR: {lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR-scheduler
    if(scheduler == "cosine_annealing"):
        print_("Setting up Cosine Annealing LR-Scheduler")
        print_(f"   Init LR: {lr}")
        print_(f"   T_max:   {scheduler_steps}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=scheduler_steps
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = IdentityScheduler(init_lr=lr)

    # seting up lr_warmup object
    lr_warmup = setup_lr_warmup(params=exp_params["training"])

    return optimizer, scheduler, lr_warmup


@log_function
def setup_lr_warmup(params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.

    Example:
    -------
        #  Learning rate is initialized with 3e-4 * (1/1000). For the first 1000 iterations
        #  or first epoch, the learning rate is updated to 3e-4 * (iter/1000).
        # after the warmup period, learning rate is fixed at 3e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
        lr_warmup = LRWarmUp(init_lr=3e-4, warmup_steps=1000, max_epochs=1)
        ...
        lr_warmup(iter=cur_iter, epoch=cur_epoch, optimizer=optimizer)  # updating lr
    """
    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if(use_warmup):
        warmup_steps = params["warmup_steps"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps)
        print_("Setting up learning rate warmup:")
        print_(f"  Target LR:     {lr}")
        print_(f"  Warmup Steps:  {warmup_steps}")
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
        print_("Not using learning rate warmup...")
    return lr_warmup



#
