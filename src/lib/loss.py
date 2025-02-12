"""
Loss functions and loss-related utils
"""

import numpy as np
import torch
import torch.nn as nn


from lib.logger import log_info, print_



class LossTracker:
    """
    Class for computing, weighting and tracking several loss functions

    Args:
    -----
    loss_params: dict
        Loss section of the experiment paramteres JSON file
    """

    def __init__(self, loss_params):
        """
        Loss tracker initializer
        """
        LOSSES = LOSS_DICT.keys()
        assert isinstance(loss_params, list), f"Loss_params must be a list, not {type(loss_params)}"
        for loss in loss_params:
            if loss["type"] not in LOSSES:
                raise NotImplementedError(f"Loss {loss['type']} not implemented. Use one of {LOSSES}")

        self.loss_computers = {}
        for loss in loss_params:
            loss_type, loss_weight = loss["type"], loss["weight"]
            self.loss_computers[loss_type] = {}
            self.loss_computers[loss_type]["metric"] = get_loss(loss_type, **loss)
            self.loss_computers[loss_type]["weight"] = loss_weight
        self.tracked_losses = list(self.loss_computers.keys())
        self.reset()
        return

    def reset(self):
        """
        Reseting loss tracker
        """
        self.loss_values = {}
        self.loss_values["_total"] = []
        self.warning = False
        return

    def __call__(self, **kwargs):
        """
        Wrapper for calling accumulate
        """
        self.accumulate(**kwargs)

    def _last_losses_to_cpu(self):
        """ Removing last losses from GPU to avoid memory leak """
        for l in self.loss_values:
            if len(self.loss_values[l]) > 0:
                self.loss_values[l][-1] = self.loss_values[l][-1].cpu()
        return

    def accumulate(self, **kwargs):
        """
        Computing the different metrics, weigting them according to their multiplier,
        and adding them to the results list.
        """
        self._last_losses_to_cpu()
        
        total_loss = 0
        for loss in self.loss_computers:
            loss_val = self.loss_computers[loss]["metric"](**kwargs)
            if isinstance(loss_val, dict):  # for loss computers that return a dictionary
                cur_total_loss = 0.
                for k, cur_loss_val in loss_val.items():
                    if k not in self.loss_values:
                        self.loss_values[k] = []
                    self.check_if_nan(loss_name=k, loss_val=cur_loss_val)
                    self.loss_values[k].append(cur_loss_val)
                    cur_total_loss = cur_total_loss + cur_loss_val
                total_loss = total_loss + cur_total_loss
            else:  # for loss computers that return a single tensor
                if loss not in self.loss_values:
                    self.loss_values[loss] = []
                self.check_if_nan(loss_name=loss, loss_val=loss_val)
                self.loss_values[loss].append(loss_val)
                total_loss = total_loss + loss_val * self.loss_computers[loss].get("weight", 1.)
        self.loss_values["_total"].append(total_loss)
        return

    def aggregate(self):
        """
        Aggregating the results for each metric
        """
        self._last_losses_to_cpu()
        self.loss_values["mean_loss"] = {}
        for loss in self.loss_values.keys():
            if loss == "mean_loss":
                continue
            self.loss_values["mean_loss"][loss] = torch.stack(self.loss_values[loss]).mean()
        self.loss_values["mean_loss"]["_total"] = torch.stack(self.loss_values["_total"]).mean()
        return

    def get_last_losses(self, total_only=False):
        """
        Fetching the last computed loss value for each loss function
        """
        if total_only:
            last_losses = self.loss_values["_total"][-1]
        else:
            last_losses = {loss: loss_vals[-1] for loss, loss_vals in self.loss_values.items()}
        return last_losses

    def summary(self, log=True, get_results=True):
        """
        Printing and fetching the results
        """
        if log:
            log_info("LOSS VALUES:")
            log_info("--------")
            for loss, loss_value in self.loss_values["mean_loss"].items():
                log_info(f"  {loss}:  {round(loss_value.item(), 5)}")

        return_val = self.loss_values["mean_loss"] if get_results else None
        return return_val
    
    def check_if_nan(self, loss_name, loss_val):
        """ Checkking if loss is NaN, and logging if so """
        is_nan_func = torch.isnan if torch.is_tensor(loss_val) else np.isnan
        if is_nan_func(loss_val) and self.warning is False:
            print_(f"WARNING! Loss {loss_name} has become NaN!...")
            self.warning = True
        return



def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    LOSSES = LOSS_DICT.keys()
    if loss_type not in LOSSES:
        raise NotImplementedError(f"Loss {loss_type} not available. Use one of {LOSSES}")

    # fetching loss class
    print(f"creating loss function of type: {loss_type}")
    loss_class = LOSS_DICT[loss_type]
    
    # handling args
    loss_params = {}
    if loss_type == "VQLoss":
        beta = kwargs.get("beta", None)
        if beta is None:
            raise ValueError(f"'beta' must be provided in order to use 'VQLoss' loss...")
        loss_params["beta"] = beta
    elif loss_type == "ActionDirKLD":
        cov = kwargs.get("cov", None)
        if cov is None:
            raise ValueError(f"'cov' must be provided in order to use 'ActionDirKLD' loss...")
        loss_params["cov"] = cov

    # instanciating loss
    loss = loss_class(**loss_params)
    return loss



class Loss(nn.Module):
    """
    Base class for custom loss functions
    """

    REQUIRED_ARGS = []

    def __init__(self):
        super().__init__()

    def _unpack_kwargs(self, **kwargs):
        """
        Fetching the required arguments from kwargs
        """
        out = []
        for arg in self.REQUIRED_ARGS:
            if arg not in kwargs:
                raise ValueError(f"Required '{arg = }' not in {kwargs.keys() = } in {self.__class__.__name__}")
            out.append(kwargs[arg])
        if len(out) == 1:
            out = out[0]
        return out



####################
# MSE-Based Losses #
####################


class MSELoss(Loss):
    """
    Overriding MSE Loss
    """

    REQUIRED_ARGS = ["pred_imgs", "target_imgs"]

    def __init__(self):
        """
        Module initializer
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Computing loss
        """
        preds, targets = self._unpack_kwargs(**kwargs)
        loss = self.mse(preds, targets)
        return loss



class PredImgMSELoss(MSELoss):
    """
    Pretty much the same MSE Loss.
    Use this loss on predicted images, while still enforcing MSELoss on predicted slots
    """
    REQUIRED_ARGS = ["pred_imgs", "target_imgs"]



class PredSlotMSELoss(MSELoss):
    """
    MSE Loss used on slot-like representations. This can be used when forecasting future slots.
    """ 
    REQUIRED_ARGS = ["preds", "targets"]



class LatentActionMSELoss(MSELoss):
    """
    Pretty much the same MSE Loss.
    Use this loss on predicted latent actions
    """
    REQUIRED_ARGS = ["pred_action_embs", "target_action_embs"]



class ActionMSELoss(MSELoss):
    """
    Pretty much the same MSE Loss.
    Use this loss on predicted actions
    """
    REQUIRED_ARGS = ["pred_actions", "target_actions"]



##############################
# KL-Divergence-Based Losses #
##############################


class ActionDirectionGaussKLD(Loss):
    """
    Kullback-Leibler Divergence loss computed on the mean and std of 
    the action direction latents.
    We enforce the distribution of action latent vectors to match N(0|C)
    """
    
    REQUIRED_ARGS = ["action_directions_dist"]
    
    def __init__(self, cov, **kwargs):
        """ """
        super().__init__()
        assert isinstance(cov, (float, int)), f"'cov' attribute must be integer or float"
        assert cov > 0, f"the diagonal covariace must be positive, not '{cov = }'!"
        self.cov = cov
        return

    def forward(self, **kwargs):
        """ Enforcing the distribution of action directions to match the base Gaussian """
        action_directions_dist = self._unpack_kwargs(**kwargs)
        mean = action_directions_dist[:, :, 0]
        std = action_directions_dist[:, :, 1]
        kld_loss = self.kld(
                mu_prior=mean,
                logvar_prior=std,
                mu_post=torch.zeros_like(mean, device=mean.device),
                logvar_post=torch.ones_like(std, device=mean.device) * self.cov,
                is_logvar=False
            ) 
        return kld_loss

    def kld(self, mu_prior, mu_post, logvar_prior, logvar_post, is_logvar=True):
        """ """
        if is_logvar:
            sigma_prior = logvar_prior.mul(0.5).exp()
            sigma_post = logvar_post.mul(0.5).exp()
        else:
            sigma_prior = logvar_prior.clone()
            sigma_post = logvar_post.clone()
            logvar_prior = torch.log(logvar_prior)
            logvar_post = torch.log(logvar_post)

        kld_1 = torch.log(sigma_prior / sigma_post)
        kld_2 = (torch.exp(logvar_post) + (mu_post - mu_prior)**2) / (2*torch.exp(logvar_prior))
        kld_loss = kld_1 + kld_2 - 1/2
        return kld_loss.mean()



#######################
# Quantization Losses #
#######################


class VQLoss(Loss):
    """
    Loss used to trained models with a catergorical (quantized) latent space
    """

    REQUIRED_ARGS = ["commit_loss", "quant_loss"]
    
    def __init__(self, beta):
        """ Loss initializition """
        super().__init__()
        self.beta = beta

    def forward(self, **kwargs):
        """
        Computing the loss
        """
        commit_loss, quant_loss = self._unpack_kwargs(**kwargs)
        loss = {
            "vq_quant_loss": quant_loss.mean(),
            "vq_commit_loss": self.beta * commit_loss.mean()
        }
        return loss



LOSS_DICT = {
    # MSE-Based
    "mse": MSELoss,
    "pred_img_mse": PredImgMSELoss,
    "pred_slot_mse": PredSlotMSELoss,
    "latent_action_mse": LatentActionMSELoss,
    "action_mse": ActionMSELoss,
    # KLD-Based Losses
    "ActionDirKLD": ActionDirectionGaussKLD,
    # Quantization Losses
    "VQLoss": VQLoss,
}

