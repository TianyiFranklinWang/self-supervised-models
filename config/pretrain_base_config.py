from abc import ABC

import torch


class PretrainBaseConfig(ABC):
    """
    Parameters used for training.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.ddp_find_unused_params = True
        self.distributed = True
        self.world_size = 1
        self.rank = 0  # global rank
        self.local_rank = 0

        self.debug = False
        self.log_path = "./logs"
        self.log_level = 'debug'
        self.save_last = True
        self.save_best = False
        self.save_interval = 50
        self.use_wandb = True
        self.wandb_log_freq = 50
        self.wandb_project_name = None
        self.wandb_exp_name = None

        self.global_seed = 42

        self.jit_fuser = 'nvfuser'
        self.model_name = None
        self.image_size = 224
        self.sync_bn = True
        self.aot_autograd = False
        self.resume = None
        self.no_resume_opt = False
        self.amp_dtype = torch.float16

        self.batch_size = 512
        self.lr = None
        self.lr_base = 1e-3
        self.lr_base_size = 512
        self.optimizer = 'sgd'
        self.momentum = 0.9
        self.weight_decay = 2e-5
        self.layer_decay = None
        self.optimizer_kwargs = {}
        self.clip_grad = None
        self.clip_mode = 'norm'

        self.scheduler = 'cosine'
        self.epochs = 300
        self.decay_epochs = 90
        self.decay_milestones = [90, 180, 270]
        self.cooldown_epochs = 0
        self.patience_epochs = 10
        self.decay_rate = 0.1
        self.min_lr = 0
        self.warmup_lr = 1e-5
        self.warmup_epochs = 5
        self.warmup_prefix = False
        self.lr_noise = None
        self.lr_noise_pct = 0.67
        self.lr_noise_std = 1.0
        self.lr_cycle_mul = 1.0
        self.lr_cycle_decay = 0.5
        self.lr_cycle_limit = 1
        self.lr_k_decay = 1.0
        self.schedule_on_updates = False

        self.data_path = None
        self.train_folder_name = 'train'
        self.mean = [0, 0, 0]
        self.std = [1, 1, 1]
        self.dataset_transform_train = None
        self.num_workers = 6
        self.pin_memory = True

        self.update_by_kwargs(**kwargs)

    def update_by_kwargs(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
