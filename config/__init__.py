from .pretrain_byol_config import *
from .pretrain_dino_config import *
from .pretrain_mae_config import *
from .registry import *


def get_config(config_name, **kwargs):
    if not is_config(config_name):
        raise RuntimeError(f'Unknown configuration ({config_name})')

    construct_fn = config_entrypoint(config_name)
    config = construct_fn(**kwargs)

    return config
