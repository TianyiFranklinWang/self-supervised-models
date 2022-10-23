import copy
import datetime
import json
import os
import shutil
import sys

import pandas as pd
import wandb


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id

    Args:
        log_path ([str]): Directory

    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = os.path.join(log_path, str(today))

    if not os.path.exists(log_today):
        os.makedirs(log_today)

    exp_id = len(os.listdir(log_today))
    log_folder = os.path.join(log_today, str(exp_id))
    log_name = today + "_" + str(exp_id)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    else:
        raise RuntimeError(f"Experiment already exists in {log_folder}")

    return log_folder, log_name


def save_config(config, log_path):
    """
    Saves a config as a json and pandas dataframe

    Args:
        config (Config): Config.
        log_path (str): Path to log folder.

    Returns:
        dict: Config as a dictionary.
    """
    dic = copy.deepcopy(config.__dict__)

    convert_config_to_string(dic)

    file_path = os.path.join(log_path, "config.json")
    with open(file_path, "w") as f:
        json.dump(dic, f)

    return dic


def convert_config_to_string(config_dict):
    for key, val in config_dict.items():
        if type(val) not in [str, bool, int, float, list, dict, type(None)]:
            config_dict[key] = str(val)
        if type(val) is dict:
            convert_config_to_string(val)


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """

    log = open(os.path.join(directory, name), "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def create_wandb_logger(config, log_folder, log_name):
    print(f"            - w&b project name: {config.wandb_project_name}")
    print(f"            - w&b experiment name: {config.wandb_exp_name}_{log_name}")

    import wandb
    wandb.init(
        project=config.wandb_project_name,
        name=f"{config.wandb_exp_name}_{log_name}",
        config=convert_config_to_string(copy.deepcopy(config.__dict__)),
        dir=log_folder,
        save_code=True,
    )


def save_metrics(epoch, epochs, lr, epoch_time, avg_loss):
    metrics = {
        'epoch': epoch + 1,
        'epochs': epochs,
        'lr': lr,
        'epoch_time': epoch_time,
        'loss': avg_loss,
    }
    return metrics


def update_history(history, metrics):
    """
    Updates a training history dataframe.

    Args:
        history (pandas dataframe or None): Previous history.
        metrics (dict): Metrics dictionary.
        epoch (int): Epoch.
        loss (float): Training loss.
        val_loss (float): Validation loss.
        time (float): Epoch duration.

    Returns:
        pandas dataframe: history
    """
    new_history = {
        "epoch": [metrics['epoch']],
        "lr": [metrics['lr']],
        "time": [metrics['epoch_time']],
        "loss": [metrics['loss']],
    }
    new_history.update(metrics)

    new_history = pd.DataFrame.from_dict(new_history)

    if history is not None:
        return pd.concat([history, new_history]).reset_index(drop=True)
    else:
        return new_history


def save_files_to_wandb(log_folder, file_names):
    if not os.path.exists(os.path.join(wandb.run.dir, log_folder)):
        os.makedirs(os.path.join(wandb.run.dir, log_folder))
    for file_name in file_names:
        shutil.copy(os.path.join(log_folder, file_name), os.path.join(wandb.run.dir, log_folder, file_name))
