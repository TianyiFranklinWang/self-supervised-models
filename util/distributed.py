import torch

from timm.utils import is_distributed_env, world_info_from_env


def init_distributed_device(config):
    config.distributed = is_distributed_env()
    if config.distributed:
        config.local_rank, _, _ = world_info_from_env()

        torch.distributed.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
        )

        config.world_size = torch.distributed.get_world_size()
        config.rank = torch.distributed.get_rank()

    if torch.cuda.is_available():
        if config.distributed:
            device = f'cuda:{config.local_rank}'
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    device = torch.device(device)
    return device


def cleanup_distributed():
    torch.distributed.destroy_process_group()


def is_global_primary(config):
    return config.rank == 0


def is_local_primary(config):
    return config.local_rank == 0


def is_primary(config, local=False):
    return is_local_primary(config) if local else is_global_primary(config)
