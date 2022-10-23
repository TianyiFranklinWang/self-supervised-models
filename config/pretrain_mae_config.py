import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from .pretrain_base_config import PretrainBaseConfig
from .registry import register_config


@register_config
class PretrainMAEConfig(PretrainBaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.wandb_project_name = "self-supervised-models"
        self.wandb_exp_name = f"pretrain_{self.model_name}"

        self.model_name = "mae_vit_base_patch16"
        self.mask_ratio = 0.75
        self.model_kwargs = {'mask_ratio': self.mask_ratio}
        self.batch_size = 8
        self.lr_base_size = 8

        self.optimizer = 'adamw'
        self.weight_decay = 0.05
        self.optimizer_kwargs = {'betas': (0.9, 0.95)}
        self.ddp_find_unused_params = False

        self.data_path = "./input/fruits-360-original-size"
        self.train_folder_name = "Training"
        self.dataset_transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)]
        )
