import copy
import os
import random

import numpy as np
import torch

from timm.models.helpers import clean_state_dict
from util.logger import convert_config_to_string


def seed_everything(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, verbose=False):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = clean_state_dict(checkpoint['state_dict'])
            if hasattr(model, 'net'):
                model.net.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            if verbose:
                print("        - Checkpoint loaded")

            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if verbose:
                    print("        - Optimizer state restored")

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])
                if verbose:
                    print("        - Loss scaler state restored")
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if verbose:
                    print(f"        - Training restored from epoch {resume_epoch}")
        else:
            if hasattr(model, 'net'):
                model.net.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            if verbose:
                print("        - Checkpoint loaded")
        return resume_epoch
    else:
        raise FileNotFoundError(f"No such checkpoint file {checkpoint_path}")


def count_parameters(model, all=False):
    """
    Count the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """

    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(config, epoch, model, loss_scaler, optimizer, log_folder, file_name):
    save_sate = {'epoch': epoch,
                 'arch': config.model_name,
                 'state_dict': get_state_dict(model, unwrap_model),
                 'optimizer': optimizer.state_dict(),
                 'config': convert_config_to_string(copy.deepcopy(config.__dict__)),
                 loss_scaler.state_dict_key: loss_scaler.state_dict()}
    save_path = os.path.join(log_folder, file_name)
    torch.save(save_sate, save_path)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    unwrapped_model = unwrap_fn(model)
    if hasattr(unwrapped_model, 'net'):
        dict = unwrapped_model.net.state_dict()
    else:
        dict = unwrapped_model.state_dict()
    return dict


def get_lr(optimizer):
    lrl = [param_group['lr'] for param_group in optimizer.param_groups]
    lr = sum(lrl) / len(lrl)
    return lr
