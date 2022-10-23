import torch


def init_distributed_device(config, rank, world_size):
    torch.distributed.init_process_group(
        backend=config.dist_backend,
        init_method=config.dist_url,
        world_size=world_size,
        rank=rank
    )

    if torch.cuda.is_available():
        device = f'cuda:{rank}'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    device = torch.device(device)
    return device


def cleanup_distributed():
    torch.distributed.destroy_process_group()
