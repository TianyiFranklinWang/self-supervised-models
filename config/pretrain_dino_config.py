import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from .pretrain_base_config import PretrainBaseConfig
from .registry import register_config


@register_config
class PretrainDINOConfig(PretrainBaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = "dino_resnet_50"
        self.byol_augment_fn = None
        self.byol_augment_fn2 = None
        self.model_kwargs = {'augment_fn': self.byol_augment_fn, 'augment_fn2': self.byol_augment_fn2,
                             'hidden_layer': 'avgpool'}
        self.batch_size = 8
        self.lr_base_size = 8

        self.wandb_project_name = "self-supervised-models"
        self.wandb_exp_name = f"pretrain_{self.model_name}"

        self.optimizer = 'adamw'
        self.weight_decay = 0.05
        self.lr_base = 3e-4
        self.optimizer_kwargs = {'betas': (0.9, 0.95)}
        self.ddp_find_unused_params = True

        self.data_path = "./input/fruits-360-original-size"
        self.train_folder_name = "Training"
        self.dataset_transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)]
        )
