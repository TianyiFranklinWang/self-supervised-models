import copy
import datetime
import json
import math
import os
import shutil
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import wandb

from timm.utils import AverageMeter


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
        if type(val) not in [str, bool, int, float, list, dict, tuple, type(None)]:
            config_dict[key] = str(val)
        if isinstance(val, dict):
            convert_config_to_string(val)


class Logger:
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

    wandb.init(
        project=config.wandb_project_name,
        name=f"{config.wandb_exp_name}_{log_name}",
        config=convert_config_to_string(copy.deepcopy(config.__dict__)),
        dir=log_folder,
        save_code=True,
    )


def save_files_to_wandb(log_folder, file_names):
    if not os.path.exists(os.path.join(wandb.run.dir, log_folder)):
        os.makedirs(os.path.join(wandb.run.dir, log_folder))
    for file_name in file_names:
        shutil.copy(os.path.join(log_folder, file_name), os.path.join(wandb.run.dir, log_folder, file_name))


class PretrainMeter:
    def __init__(self, log_folder=None):
        self.timers = {}

        self.best_loss = math.inf
        self.metrics = None

        self.history = None

        self.log_folder = log_folder

    def start_timer(self, timer_name):
        self.timers[timer_name] = time.time()

    def stop_timer(self, timer_name):
        self.timers[timer_name] = time.time() - self.timers[timer_name]

    def init_metrics(self, epoch, epochs):
        self.metrics = {
            'epoch': epoch + 1,
            'epochs': epochs,
            'avg_loss': AverageMeter(),
        }

    def update_metrics(self, lr, timer_name):
        self.metrics['lr'] = lr
        self.metrics['epoch_time'] = self.timers[timer_name]
        self.metrics['avg_loss'] = self.metrics['avg_loss'].avg

    def print_metrics(self):
        print(
            f"Epoch {self.metrics['epoch']:02d}/{self.metrics['epochs']:02d} \t"
            f" lr={self.metrics['lr']:.1e}\t"
            f" t={self.metrics['epoch_time']:.2f}s\t"
            f" loss={self.metrics['avg_loss']:.3f}",
            end=''
        )

    def update_history(self):
        new_history = {
            "epoch": [self.metrics['epoch']],
            "lr": [self.metrics['lr']],
            "time": [self.metrics['epoch_time']],
            "loss": [self.metrics['avg_loss']],
        }

        new_history = pd.DataFrame.from_dict(new_history)

        if self.history is not None:
            self.history = pd.concat([self.history, new_history]).reset_index(drop=True)
        else:
            self.history = new_history

    def plot_dashboard(self, x, y, title):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.history[x],
                y=self.history[y],
                mode='lines',
            )
        )
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font_size=32,
            autosize=False,
            width=1920,
            height=1080,
            font=dict(
                family="Times New Roman",
                size=20,
            )
        )
        fig.update_xaxes(title_text=x)
        fig.update_yaxes(title_text=y)
        fig.write_image(os.path.join(self.log_folder, f"{title.replace(' ', '_')}.png"))
