import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

def get_optimizer(optimizer_name):
    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW
    elif optimizer_name.lower() == 'adam':
        return torch.optim.Adam
    elif optimizer_name.lower() == 'adamax':
        return torch.optim.Adamax


def get_lr_scheduler(scheduler_name):
    if scheduler_name.lower() == 'lrdrop':
        return torch.optim.lr_scheduler.ReduceLROnPlateau
    elif scheduler_name.lower() == 'cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_name.lower() == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR
    elif scheduler_name.lower() == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR
    elif scheduler_name.lower() == 'constant':
        return torch.optim.lr_scheduler.ConstantLR


def get_lr_scheduler_params(scheduler_name, **kwargs):

    if scheduler_name.lower() == 'lrdrop':
        params = kwargs
        interval = 'epoch'
        return params, interval

    elif scheduler_name.lower() in {'cos', 'onecycle'}:
        params = kwargs
        interval = 'step'
        return params, interval

    elif scheduler_name.lower() == 'multistep':
        interval = 'epoch'
        params = kwargs
        return params, interval

    elif scheduler_name.lower() == 'lambdalr':
        interval = 'epoch'
        params = kwargs
        return params, interval
    elif scheduler_name.lower() == 'constant':
        params = kwargs
        interval = 'step'
        return params, interval
