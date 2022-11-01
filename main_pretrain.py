import os

import torch

import config
from train import pretrain_train
from util.distributed import cleanup_distributed, init_distributed_device, is_primary
from util.logger import create_logger, create_wandb_logger, prepare_log_folder, save_config

CONFIG_NAME = "PretrainBYOLConfig"
GLOBAL_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

if __name__ == '__main__':
    if GLOBAL_RANK == 0:
        print("\n -> Initializing training protocol")

    if GLOBAL_RANK == 0:
        print(f"    - Loading configuration from {CONFIG_NAME}")
    config = config.get_config(CONFIG_NAME, log_level='user')

    if GLOBAL_RANK == 0:
        print(f"    - Initializing process group with world size of {WORLD_SIZE}")
    device = init_distributed_device(config)
    if config.log_level == 'debug':
        print(f"        - Initialized on rank{config.local_rank}")
    assert config.rank >= 0

    if config.distributed:
        torch.distributed.barrier()

    log_folder = None
    if is_primary(config):
        if not config.debug:
            print("    - Initializing logging system")
            log_folder, log_name = prepare_log_folder(config.log_path)
            if config.log_level == 'debug':
                print(f"        - Logging results to {log_folder}")
            config_dict = save_config(config, log_folder)
            create_logger(directory=log_folder, name='logs.txt')
            if config.use_wandb:
                if config.log_level == 'debug':
                    print("        - Enabling w&b logging system")
                create_wandb_logger(config, log_folder, log_name)

    pretrain_train(config, device=device, log_folder=log_folder)

    if config.distributed:
        torch.distributed.barrier()
        cleanup_distributed()
